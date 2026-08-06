from __future__ import annotations
from dataclasses import asdict,dataclass
from .neural_echo import require_torch


@dataclass(frozen=True)
class CodecCopyConfig:
    codebooks:int=8
    vocabulary_size:int=1024
    hidden_dim:int=192
    understanding_layers:int=4
    attention_heads:int=4
    read_sigma:float=0.65


def create_codec_copy_model(config=CodecCopyConfig()):
    torch,nn,functional=require_torch(); dim=config.hidden_dim
    class CodecCopyModel(nn.Module):
        def __init__(self):
            super().__init__(); self.config=config
            self.embeddings=nn.ModuleList([nn.Embedding(config.vocabulary_size,dim) for _ in range(config.codebooks)])
            self.silence=nn.Parameter(torch.zeros(dim)); self.layer_embedding=nn.Parameter(torch.randn(config.understanding_layers,dim)*.02)
            self.layer_attention=nn.ModuleList([nn.MultiheadAttention(dim,config.attention_heads,batch_first=True) for _ in range(config.understanding_layers)])
            self.optimizers=nn.ModuleList([nn.GRUCell(dim*2,dim) for _ in range(config.understanding_layers)])
            self.controller=nn.GRUCell(dim*3,dim)
            self.readiness=nn.Sequential(nn.LayerNorm(dim*2),nn.Linear(dim*2,1))
            self.advance=nn.Sequential(nn.LayerNorm(dim*3),nn.Linear(dim*3,1),nn.Softplus())
            self.voice_head=nn.Sequential(nn.LayerNorm(dim*3),nn.Linear(dim*3,1))
            self.output_heads=nn.ModuleList([nn.Linear(dim*3,config.vocabulary_size) for _ in range(config.codebooks)])

        def _embed(self,codes):
            valid=codes.ge(0); clamped=codes.clamp_min(0)
            parts=[embedding(clamped[...,index])*valid[...,index:index+1] for index,embedding in enumerate(self.embeddings)]
            return sum(parts)/valid.sum(-1,keepdim=True).clamp_min(1),valid.any(-1)

        def forward(self,inputs,ablate_tape=False,ablate_understanding=False):
            embedded,input_valid=self._embed(inputs); batch,ticks,_=embedded.shape
            positions=torch.arange(ticks,device=inputs.device,dtype=embedded.dtype)
            layers=[embedded.new_zeros(batch,dim) for _ in range(config.understanding_layers)]
            controller=embedded.new_zeros(batch,dim); read_phase=embedded.new_zeros(batch)
            logits=[]; voices=[]; phases=[]; readiness_history=[]
            visible=input_valid.cumsum(1)
            for tick in range(ticks):
                current=torch.where(input_valid[:,tick:tick+1],embedded[:,tick],self.silence.unsqueeze(0))
                previous=layers; updated=[]
                for level,(attention,optimizer) in enumerate(zip(self.layer_attention,self.optimizers)):
                    memory=torch.stack([*previous,*updated],1); query=(previous[level]+self.layer_embedding[level]).unsqueeze(1)
                    context=attention(query,memory,memory,need_weights=False)[0][:,0]
                    lower=current if level==0 else updated[level-1]
                    updated.append(optimizer(torch.cat([lower,context],-1),previous[level]))
                layers=updated; top=layers[-1]
                available=positions.unsqueeze(0)<visible[:,tick:tick+1]
                weight=torch.exp(-.5*((positions.unsqueeze(0)-read_phase.unsqueeze(1))/config.read_sigma)**2)*available
                weight=weight/weight.sum(1,keepdim=True).clamp_min(1e-6)
                tape_context=torch.einsum("bt,btd->bd",weight,embedded)
                if ablate_tape:tape_context=tape_context*0
                if ablate_understanding:top=top*0
                controller=self.controller(torch.cat([current,tape_context,top],-1),controller)
                decoded=torch.cat([controller,tape_context,top],-1)
                ready=self.readiness(torch.cat([controller,top],-1)).sigmoid().squeeze(-1)
                step=self.advance(decoded).squeeze(-1)*ready
                remaining=(visible[:,tick]-read_phase).sigmoid(); read_phase=read_phase+step*remaining
                logits.append(torch.stack([head(decoded) for head in self.output_heads],1)); voices.append(self.voice_head(decoded).squeeze(-1)); phases.append(read_phase); readiness_history.append(ready)
            diagnostics={"read_phase":torch.stack(phases,1),"readiness":torch.stack(readiness_history,1),"understanding":torch.stack(layers,1)}
            return torch.stack(logits,1),torch.stack(voices,1),diagnostics
        def export_config(self):return asdict(config)
    return CodecCopyModel()
