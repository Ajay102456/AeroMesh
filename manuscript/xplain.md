# AeroMesh, explained simply

## The problem

When a big typhoon or earthquake hits, cell towers and fiber lines get knocked out. Rescue teams on the ground lose phone and internet service exactly when they need it most. Think Typhoon Odette hitting the Philippines in 2021, or the 2018 earthquake and tsunami in Palu, Indonesia. Roads get blocked, people cannot reach the area for days, and nobody can even see how bad the damage is in real time.

## The idea

Instead of waiting for engineers to drive out and fix cell towers, send a swarm of drones to form a temporary flying internet network over the disaster zone right away. Picture a mesh Wi-Fi system, but each "router" is a drone hovering in the air. Some drones are called relay drones and their only job is to pass signal around, like repeaters. Other drones are inference drones and they carry a small onboard AI model that looks at camera images and tries to judge how badly buildings are damaged, without needing to send huge video files back to a command center over a weak connection.

## Why this is actually hard, not just "put drones in the sky"

1. **Drones die.** Battery runs out, GPS gets jammed, one crashes. The network has to notice a hole opened up and automatically reposition other drones to patch it, with no human telling them where to go. This is called self-healing.
2. **Not everyone deserves equal bandwidth.** A rescue team pulling someone out of rubble needs the connection far more than a civilian trying to check social media. The system automatically prioritizes using an auction system, similar to eBay bidding, where drones sell their limited bandwidth to the task that needs it most.
3. **The onboard AI has to know when it is unsure.** A tiny drone computer cannot run a giant, maximally accurate AI model, so it runs something smaller and faster instead. The cheap model handles easy cases itself, but when an image is confusing because of smoke, a weird angle, or debris it has never seen, it says it is not confident and escalates that image to a more powerful drone instead of silently guessing wrong. This uncertainty-aware behavior is a safety mechanism against an AI that is confidently wrong during a life-or-death assessment.

## What makes it Southeast Asia specific

Instead of a generic drones-in-disasters paper, it is grounded in three real documented events with real regulatory rules.

- **Philippines**, Typhoon Odette knocking out telecom, under actual Philippine aviation drone rules from CAAP.
- **Indonesia**, the Palu earthquake and tsunami, where the islands themselves make ground access brutal, under Indonesian aviation rules from DGCA.
- **Vietnam**, Typhoon Yagi flooding the Mekong region in 2024.

This matters because drone laws and disaster geography differ in every country, so a system that ignores that difference is not realistic.

## What was actually done

1. Took an existing rough research concept and expanded it into a full academic manuscript targeting the journal Drones, published by MDPI.
2. Wrote a formal math problem statement, defining things like how connected the drone network is using graph theory, drawing on an idea from a 1973 paper called algebraic connectivity.
3. Specified two algorithms, one for self-healing mesh repair and one for the priority auction.
4. Wrote a full experiment plan, including which public disaster-imagery datasets to test on, six in total, including a Southeast-Asia-specific one called AINDAA, what to compare against, and which ablation studies to run, meaning turning off one piece at a time to see what breaks.
5. Checked 76 real references one by one against actual published papers, with nothing invented. A few original citations in the source document were flat-out wrong, with incorrect authors or unverifiable papers, and those were corrected or dropped rather than kept.
6. Built diagrams of the system architecture and the three disaster scenarios as vector graphics, delivered in both PDF and SVG.
7. Compiled everything into a 24-page LaTeX document using the real MDPI journal template, with a matching blind, anonymous version for peer review.

## The most important finding, and it is an unusual one

There is no "it worked, here is the data" result yet. The Results section is intentionally left as labeled placeholders, not because anything failed, but because no actual simulation has been run. This was a deliberate choice, since the plan is for the real simulation to be run later. The honest approach was to write the complete, pre-registered experimental plan, specifying exactly what will be measured and how, without faking any numbers to make the paper look finished. That is a mark of good scientific practice rather than a weakness, since papers that announce results before the experiment exists are the ones that get retracted.

## Bottom line

AeroMesh is a plan for a drone swarm that turns itself into emergency internet plus an AI damage scanner after a disaster, built specifically around real Southeast Asian disaster corridors and real regulations, with the full math, algorithms, and experiment design written up and verified, waiting for the simulation to be run so the real numbers can be filled in.
