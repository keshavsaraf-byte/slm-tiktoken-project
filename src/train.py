import argparse, math, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.config import load_config
from src.dataset import TokenStreamDataset
from src.model import SmallLanguageModel
from src.tokenizer import TikTokenTokenizer

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config", default="configs/development.yaml")
    a=p.parse_args(); c=load_config(a.config)
    random.seed(c.get("seed",42)); torch.manual_seed(c.get("seed",42))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(c.get("seed",42))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=TikTokenTokenizer(c["tokenizer"].get("encoding_name","cl100k_base"))
    dp=c["data"]; trp, vap=Path(dp["train_path"]), Path(dp["val_path"])
    if not trp.exists() or not vap.exists():
        raise FileNotFoundError("Run python -m src.prepare_data --config " + a.config)
    train=torch.load(trp,map_location="cpu",weights_only=True)
    val=torch.load(vap,map_location="cpu",weights_only=True)
    mc, tc=c["model"],c["training"]; ctx=int(mc["context_length"])
    td=TokenStreamDataset(train,ctx); vd=TokenStreamDataset(val,ctx)
    if not len(td): raise ValueError("Not enough train tokens; add text or reduce context_length")
    if not len(vd): vd=td
    loader=DataLoader(td,batch_size=int(tc["batch_size"]),shuffle=True,drop_last=False)
    vloader=DataLoader(vd,batch_size=int(tc["batch_size"]),shuffle=False)
    model=SmallLanguageModel(tok.vocab_size,ctx,int(mc["d_model"]),int(mc["n_heads"]),
        int(mc["n_layers"]),float(mc["dropout"]),bool(mc.get("tie_embeddings",True))).to(device)
    print(f"Device: {device} | vocab: {tok.vocab_size:,} | parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt=torch.optim.AdamW(model.parameters(),lr=float(tc["learning_rate"]),weight_decay=float(tc.get("weight_decay",.1)))
    best=float("inf"); out=Path(tc["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    it=iter(loader); steps=int(tc["max_steps"]); interval=int(tc["eval_interval"])
    @torch.no_grad()
    def evaluate():
        model.eval(); ls=[]
        for j,(x,y) in enumerate(vloader):
            if j>=int(tc.get("eval_batches",10)): break
            _,l=model(x.to(device),y.to(device)); ls.append(l.item())
        model.train(); return sum(ls)/max(1,len(ls))
    for step in range(1,steps+1):
        try: x,y=next(it)
        except StopIteration: it=iter(loader); x,y=next(it)
        progress=step/steps; lo=float(tc.get("min_learning_rate",tc["learning_rate"])); hi=float(tc["learning_rate"])
        lr=lo+.5*(hi-lo)*(1+math.cos(math.pi*progress))
        for g in opt.param_groups: g["lr"]=lr
        model.train(); x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
        _,loss=model(x,y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step==1 or step%interval==0 or step==steps:
            vl=evaluate(); print(f"step {step}/{steps} | train {loss.item():.4f} | val {vl:.4f} | lr {lr:.2e}")
            if vl<best:
                best=vl; torch.save({"model_state_dict":model.state_dict(),"model_config":{
                    "vocab_size":tok.vocab_size,"context_length":ctx,"d_model":int(mc["d_model"]),
                    "n_heads":int(mc["n_heads"]),"n_layers":int(mc["n_layers"]),"dropout":float(mc["dropout"]),
                    "tie_embeddings":bool(mc.get("tie_embeddings",True))},
                    "encoding_name":tok.encoding_name,"step":step,"val_loss":vl},out/"best.pt")
    print(f"Done. Best validation loss {best:.4f}. Checkpoint: {out/'best.pt'}")
if __name__ == "__main__": main()
