from __future__ import annotations
from dataclasses import asdict,dataclass
from .neural_echo import require_torch


@dataclass(frozen=True)
class CodecCopyConfig:
    codebooks:int=8
    vocabulary_size:int=1024
    hidden_dim:int=256
    understanding_layers:int=4
    dropout:float=0.1


def create_codec_copy_model(config=CodecCopyConfig()):
    """A fully learned streaming listen-then-repeat network.

    The model never indexes or returns an input codec token.  Past audio is represented
    only by learned recurrent activations.  A decoder uses soft attention over those
    activations and its own previous acoustic output to generate every codec token.
    """
    torch,nn,_=require_torch(); dim=config.hidden_dim

    class CodecCopyModel(nn.Module):
        def __init__(self):
            super().__init__(); self.config=config
            self.input_embeddings=nn.ModuleList([nn.Embedding(config.vocabulary_size,dim) for _ in range(config.codebooks)])
            self.output_embeddings=nn.ModuleList([nn.Embedding(config.vocabulary_size,dim) for _ in range(config.codebooks)])
            self.no_audio=nn.Parameter(torch.zeros(dim)); self.no_output=nn.Parameter(torch.zeros(dim))
            self.encoder_cells=nn.ModuleList([nn.GRUCell(dim,dim) for _ in range(config.understanding_layers)])
            self.encoder_norms=nn.ModuleList([nn.LayerNorm(dim) for _ in range(config.understanding_layers)])
            self.memory_keys=nn.Linear(dim,dim,bias=False); self.memory_values=nn.Linear(dim,dim,bias=False)
            self.decoder=nn.GRUCell(dim*3,dim)
            self.query=nn.Linear(dim,dim,bias=False)
            self.voice_head=nn.Sequential(nn.LayerNorm(dim*2),nn.Linear(dim*2,1))
            self.output_heads=nn.ModuleList([nn.Sequential(nn.LayerNorm(dim*2),nn.Linear(dim*2,config.vocabulary_size)) for _ in range(config.codebooks)])
            self.dropout=nn.Dropout(config.dropout)

        @staticmethod
        def _masked_mean(parts,valid):
            return sum(parts)/valid.sum(-1,keepdim=True).clamp_min(1)

        def _input_embedding(self,codes):
            valid=codes.ge(0); safe=codes.clamp_min(0)
            parts=[table(safe[...,book])*valid[...,book:book+1] for book,table in enumerate(self.input_embeddings)]
            return self._masked_mean(parts,valid),valid.any(-1)

        def _output_embedding(self,codes):
            valid=codes.ge(0); safe=codes.clamp_min(0)
            parts=[table(safe[...,book])*valid[...,book:book+1] for book,table in enumerate(self.output_embeddings)]
            return self._masked_mean(parts,valid),valid.any(-1)

        def forward(self,inputs,teacher_tokens=None,teacher_ratio=0.0):
            encoded,input_valid=self._input_embedding(inputs); batch,ticks,_=encoded.shape
            states=[encoded.new_zeros(batch,dim) for _ in range(config.understanding_layers)]
            memories=[]
            for tick in range(ticks):
                value=torch.where(input_valid[:,tick:tick+1],encoded[:,tick],self.no_audio.unsqueeze(0))
                for level,(cell,norm) in enumerate(zip(self.encoder_cells,self.encoder_norms)):
                    proposal=cell(value,states[level]); states[level]=norm(states[level]+self.dropout(proposal))
                    value=states[level]
                memories.append(states[-1])
            memory=torch.stack(memories,1); keys=self.memory_keys(memory); values=self.memory_values(memory)
            decoder=encoded.new_zeros(batch,dim); previous=self.no_output.unsqueeze(0).expand(batch,-1)
            logits=[]; voices=[]; attention_history=[]
            teacher_embedded=teacher_valid=None
            if teacher_tokens is not None: teacher_embedded,teacher_valid=self._output_embedding(teacher_tokens)
            scale=dim**-.5
            for tick in range(ticks):
                # Only learned encoder activations from audio observed up to this tick are visible.
                visible=torch.arange(ticks,device=inputs.device).unsqueeze(0)<=tick
                score=torch.einsum('bd,btd->bt',self.query(decoder),keys)*scale
                attention=score.masked_fill(~visible,-1e4).softmax(-1)
                context=torch.einsum('bt,btd->bd',attention,values)
                current=torch.where(input_valid[:,tick:tick+1],encoded[:,tick],self.no_audio.unsqueeze(0))
                decoder=self.decoder(torch.cat([current,context,previous],-1),decoder)
                decoded=torch.cat([decoder,context],-1)
                step_logits=torch.stack([head(decoded) for head in self.output_heads],1)
                logits.append(step_logits); voices.append(self.voice_head(decoded).squeeze(-1)); attention_history.append(attention)
                predicted=torch.stack([table(step_logits[:,book].softmax(-1)) for book,table in enumerate(self.output_embeddings)],0).mean(0)
                previous=predicted
                if teacher_embedded is not None and tick+1<ticks and teacher_ratio>0:
                    use_teacher=(torch.rand(batch,device=inputs.device)<teacher_ratio)&teacher_valid[:,tick]
                    previous=torch.where(use_teacher.unsqueeze(-1),teacher_embedded[:,tick],previous)
            diagnostics={"attention":torch.stack(attention_history,1),"understanding":torch.stack(states,1)}
            return torch.stack(logits,1),torch.stack(voices,1),diagnostics

        def export_config(self): return asdict(config)
    return CodecCopyModel()
