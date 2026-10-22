import argparse, torch
from src.config import load_config
from src.model import SmallLanguageModel
from src.tokenizer import TikTokenTokenizer

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature, top_k, eot_id):
    model.eval()
    for _ in range(max_new_tokens):
        logits,_=model(idx[:,-model.context_length:])
        logits=logits[:,-1,:]/max(float(temperature),1e-5)
        if top_k and int(top_k)>0:
            k=min(int(top_k),logits.size(-1)); vals,_=torch.topk(logits,k)
            logits[logits<vals[:,-1,None]]=-float("inf")
        nxt=torch.multinomial(torch.softmax(logits,dim=-1),1)
        idx=torch.cat((idx,nxt),dim=1)
        if int(nxt.item())==int(eot_id): break
    return idx

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",default="checkpoints/best.pt")
    p.add_argument("--config",default="configs/development.yaml"); p.add_argument("--prompt",default="Hello")
    p.add_argument("--max-new-tokens",type=int,default=None); a=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(a.checkpoint,map_location=device,weights_only=False)
    tok=TikTokenTokenizer(ck.get("encoding_name","cl100k_base"))
    model=SmallLanguageModel(**ck["model_config"]).to(device); model.load_state_dict(ck["model_state_dict"])
    ids=tok.encode(a.prompt) or [tok.eot_id]; idx=torch.tensor([ids],dtype=torch.long,device=device)
    gc=load_config(a.config).get("generation",{})
    result=generate(model,idx,a.max_new_tokens or int(gc.get("max_new_tokens",80)),
                    gc.get("temperature",.8),gc.get("top_k",40),tok.eot_id)
    print(tok.decode(result[0].tolist()))
if __name__ == "__main__": main()
