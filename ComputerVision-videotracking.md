# Video Tracking

## [2011] Video Tracking : theory and practise 
### What is video tracking
Definition: The process of estimating over time the location of one or more objects using a camera is referred to as video tracking.
- Chanllenges: clutter(杂乱背景), change in pose， Ambient illumination, Noise, Occlusions(partial and total)
  - typical motion behavior
  - pre-existing occlusion patterns
  - high-level reasoning method
  - multi-hypothesis method
  - propagate tracking hypothesis
- Components
  - extract object features
  - target representation
  - propagation
  - track management:stragegy to manage targets appearing and disappearing.
    - target disappear: terminate trajectory
    - target birth: initialise a new trajectory
    - track loss

