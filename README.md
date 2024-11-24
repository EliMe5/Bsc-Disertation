# BSc Dissertation - Breaking Omnifold

This repository contains the code used in the dissertation **Breaking Omnifold** by Eliott Menard. It includes:

1. **Data Generation Code**: 
   - Generates synthetic datasets as described in the dissertation.
   - Features detailed annotations for ease of understanding and reproducibility.
   - Note: The T-SNE visualizations are currently incomplete and may not provide meaningful insights due to time constraints.

2. **Modified Omnifold Implementation**:
   - Based on the original Omnifold code from [hep-lbdl/OmniFold](https://github.com/hep-lbdl/OmniFold).
   - Incorporates the modifications discussed in the dissertation for re-weighting.
   - Includes an additional change aimed at optimizing resource usage for lower-end devices, albeit with some trade-offs in performance.
   - Implements an iterative trial framework for training neural networks, as outlined in the dissertation.
   - Note: This code lacks thorough documentation, and the implemented changes may not be immediately apparent.

---

## Notes:
- The data generation code is well-annotated and functional.
- The Omnifold code modifications, while functional, are not properly commented. Additional work may be needed to fully understand the changes.

---

## References:
- The original Omnifold implementation can be found at [hep-lbdl/OmniFold](https://github.com/hep-lbdl/OmniFold).
- For more details on the Omnifold methodology, refer to the official paper: [OmniFold: A Method to Simultaneously Unfold All Observables](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.182001), published in *Physical Review Letters*.

Feel free to reach out or open an issue if you have questions or suggestions for improvements!
