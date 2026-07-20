"""Behavioral evaluation of a trained model. Run: python scratch_eval.py [model.zip] [steps]"""
import sys, numpy as np
from stable_baselines3 import PPO
from snake_rl.train import build_vec
from snake_rl.watch import _world_of

def main(model_path="models/snake.zip", nsteps=8000):
    norm = model_path.rsplit("/",1)[0] + "/vecnormalize.pkl" if "/" in model_path else "vecnormalize.pkl"
    model = PPO.load(model_path, device="cpu")
    vec = build_vec(1, 777, training=False, norm_path=norm)
    obs = vec.reset()
    eaten=act_dash=steps=ep=0; deaths={"obstacle":0,"self":0}; eplens=[]; stam=[]; energy=[]; cur=0
    for _ in range(nsteps):
        a,_ = model.predict(obs, deterministic=False)
        w=_world_of(vec); prev=w.stamina; stam.append(prev); energy.append(w.energy)
        obs,_,done,infos = vec.step(a)
        if not done[0]: act_dash += int(_world_of(vec).stamina < prev)
        eaten += infos[0].get("ate",0); steps+=1; cur+=1
        if done[0]:
            dc = infos[0].get("death_cause")
            if dc in deaths: deaths[dc]+=1
            eplens.append(cur); cur=0; ep+=1; obs=vec.reset()
    s=np.array(stam); e=np.array(energy)
    print(f"model: {model_path}  ({steps} steps, {ep} episodes)")
    print(f"  catch rate:   {eaten/steps*1000:5.1f} chickens / 1000 steps  (total {eaten})")
    print(f"  dash usage:   {act_dash/steps*100:5.1f}% of steps actually dashing")
    print(f"  stamina:      mean {s.mean():4.1f}/{30}  std {s.std():4.1f}  (cycles: frac>10 {(s>10).mean():.2f}, frac>20 {(s>20).mean():.2f})")
    print(f"  energy:       mean {e.mean():5.1f}/100")
    print(f"  episode len:  mean {np.mean(eplens):5.0f} steps" if eplens else "  (no completed episodes)")
    print(f"  deaths:       obstacle {deaths['obstacle']}, self {deaths['self']}  ({(deaths['obstacle']+deaths['self'])/max(steps,1)*1000:.1f}/1000 steps)")
    vec.close()

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv)>1 else "models/snake.zip"
    n = int(sys.argv[2]) if len(sys.argv)>2 else 8000
    main(m, n)
