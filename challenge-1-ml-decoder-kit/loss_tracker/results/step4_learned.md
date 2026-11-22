What this says:

- **The network does not beat the model of step 2**, so it does not earn its place and was not decoded. On
  the 4,000 validation shots: exact round 48.2% against 50.9%; within one round 77.4% against 79.3%; mean log
  P(true round) −1.390 against −1.325 for the frozen version. It is level with the version that drops shared
  checks (−1.468) and below the coupled versions.
- **Where it loses and where it wins.** On atoms that share no check with another flagged atom, the network is
  slightly better than the model (mean log P −1.234 against −1.271; within one round 82.2% against 80.7%). So the
  hand-built emission model leaves a little on the table: the network can use the link between an atom's X and Z
  triples, which the model ignores. On atoms that share X and Z checks with other flagged atoms (54% of all), it
  is clearly worse (−1.485 against −1.362). It sees only how many other flagged atoms sit on each check, not when
  they were lost, and the model's coupling of the atoms is worth more than the network's flexibility.
- A next step would combine the two: the network's per-atom evidence inside the model's coupling. It is not
  built here.
- Training took 547 s on 4 CPU threads (20 epochs; the mean cross-entropy over the 72 slots fell from 3.95 to
  2.84 nats).
