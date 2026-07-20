import argparse
from .train import train
from .watch import run_watch


def main():
    p = argparse.ArgumentParser(prog="snake_rl")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=2_000_000)
    t.add_argument("--envs", type=int, default=8)
    t.add_argument("--model", default="models/snake.zip")
    t.add_argument("--reset", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    w = sub.add_parser("watch")
    w.add_argument("--model", default="models/snake.zip")
    w.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    if a.cmd == "train":
        train(a.steps, a.envs, a.model, a.reset, a.seed)
    else:
        run_watch(a.model, a.seed)


if __name__ == "__main__":
    main()
