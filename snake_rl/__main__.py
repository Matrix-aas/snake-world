import argparse
from .train import train
from .watch import run_watch, run_headless


def main():
    p = argparse.ArgumentParser(prog="snake_rl")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=2_000_000)
    t.add_argument("--envs", type=int, default=8)
    t.add_argument("--model", default="models/snake.zip")
    t.add_argument("--reset", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--save-every", type=int, default=50_000)
    t.add_argument("--log-every", type=int, default=10_000)
    w = sub.add_parser("watch")
    w.add_argument("--model", default="models/snake.zip")
    w.add_argument("--seed", type=int, default=None)
    w.add_argument("--sim-hz", type=int, default=10, help="simulation steps per second (lower = slower)")
    w.add_argument("--headless", action="store_true", help="run N episodes without a window and print stats")
    w.add_argument("--episodes", type=int, default=5)
    a = p.parse_args()
    if a.cmd == "train":
        train(a.steps, a.envs, a.model, a.reset, a.seed, a.save_every, a.log_every)
    elif a.headless:
        run_headless(a.model, a.seed, a.episodes)
    else:
        run_watch(a.model, a.seed, sim_hz=a.sim_hz)


if __name__ == "__main__":
    main()
