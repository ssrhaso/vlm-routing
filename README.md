# Efficient Vision-Language Model Routing

> **Make VLMs 3-5x faster by routing simple queries to CLIP, complex queries to LLaVA.**

##  The Problem
- **LLaVA:** 1500ms (Slow & Expensive)
- **CLIP:** 50ms (Fast & Cheap)
- **Reality:** 80% of user queries are simple ("What color is this?").
- **Solution:** A lightweight router (18ms) that sends simple queries to the fast model.