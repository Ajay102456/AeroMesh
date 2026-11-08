#!/usr/bin/env python3
import sys, csv

def summarize(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["dstPort"]) in (5000, 5001, 5002):
                rows.append(r)
    if not rows:
        print(f"{path}: no application flows recorded")
        return
    by_cls = {5000: "responder", 5001: "medical", 5002: "civilian"}
    for port, name in by_cls.items():
        sub = [r for r in rows if int(r["dstPort"]) == port]
        tx = sum(int(r["txPackets"]) for r in sub)
        rx = sum(int(r["rxPackets"]) for r in sub)
        delay = [float(r["meanDelayMs"]) for r in sub if float(r["meanDelayMs"]) > 0]
        pdr = rx / tx if tx else 0.0
        meandelay = sum(delay) / len(delay) if delay else 0.0
        print(f"{name:10s} tx={tx:6d} rx={rx:6d} pdr={pdr:.3f} meanDelayMs={meandelay:.1f}")

if __name__ == "__main__":
    summarize(sys.argv[1])
