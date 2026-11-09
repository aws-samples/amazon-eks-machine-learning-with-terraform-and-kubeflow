---
description: Price a fleet offline — memory floors, minimal degrees, the baseline and whether any co-resident placement exists. No cluster.
argument-hint: <fleets/name.yaml>
allowed-tools: [Bash, Read, Grep]
---

From the repository's `tutorials/multi-model-gpu-sharing` directory:

```bash
PYTHONPATH=src python3 -m colocation.agent plan --fleet $ARGUMENTS
```

Offline, no node captured. Run it **every time anything changes** — a fleet file, a profile round,
a store — not once per fleet.

Read the output in this order and stop at the first thing that refuses:

| what you see | what it means | what to do |
|---|---|---|
| `the hardware store holds no shape for '<type>'`, or `nobody has observed what a <type> presents` | nobody has leased this instance type, so the store has no address for its measurements | `plan` cannot fix this, not even with `--gpu-count`: only a node can be observed. `/fleet-profile <fleet> --gpu-count N` holds one node, files what it reports, measures nothing and exits 0; then run the same command again, and `plan` resolves offline from then on. Needed once per type. Do not edit the fleet file around it |
| `no environment has measured how much memory a <product> has` | the card's memory total has never been measured, and there is no default | on a type nobody has leased, the first `profile` or `search` files it off the node — say so and go on. On a type that has been run, the store or the `--gpu-product` is wrong: **stop**. Never supply the number yourself |
| `this fleet cannot be placed on this node` (exit 1) | a tenant's KV floor exceeds one GPU at every degree | **stop.** Name the tenant. The fleet file has to change — a lower rate, or one fewer tenant |
| `NO CO-RESIDENT PLACEMENT IS FEASIBLE` | no two tenants' minimum envelopes fit on one GPU | **stop**, unless the next line is `BUT NOT ON THE WEIGHTS`. Co-location has nothing to sell this fleet. Report the closest pair and by how much it is over |
| `BUT NOT ON THE WEIGHTS` (after the line above) | some pairs fit on weights alone; only the computed KV, activation and reserve terms rule them out | go on: `search` warns and runs anyway, because those terms have been measured wrong before. Profile the pair's tenants first |
| `THE BASELINE ABOVE IS AN UPPER BOUND` | a tenant met its target at the degree shown, but a narrower degree was never measured | note it: a later `profile` round can still shrink the baseline |
| `BASELINE NOT ESTABLISHED` | normal on a fresh fleet — every degree is a memory floor | go on to `/fleet-sweep` |
| `<-- FLOOR, not a measured minimum` | which tenants still need a solo round | this is the `/fleet-profile` list |

A fleet with `hardware.max_nodes` above 1 also prints `--- splits over up to N nodes ---`: each
tenant on one node, ranked by fewest nodes, then GPUs, then margin, with the first split placed node
by node. `Disjoint over nodes ... more than the N this fleet allows` gates nothing: the splits share
GPUs, so a fleet that needs more nodes disjoint is still searched. `this fleet cannot be split over
N node(s): the weights alone rule every split out` (exit 1) is a **stop**. `NO SPLIT OVER N NODE(S)
IS PROVEN INFEASIBLE` is not a proof: profile first. `<tenant> fits no node even alone` names the
tenant to change.

**Never quote `sum(floor)` as the baseline, or compute a saving against it.** It is printed as a
lower bound because a floor that fits may still miss its target, and then the tenant needs more
shards and the baseline goes **up**. A saving quoted against the floor sum is an overstatement, and
it is the single error the whole design exists to prevent.

The placements are on the *shape* of the instance type. They name GPU ordinals and look deployable.
They are not.
