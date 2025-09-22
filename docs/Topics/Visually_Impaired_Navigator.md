# Visually Impaired Navigator

One of the most significant and ongoing daily challenges for people with vision impairments is navigating new or busy environments.  Current assistive technologies, such sonar-equipped canes, AI-powered smart glasses, and GPS-based navigation systems, frequently struggle with real-time obstacle avoidance and environment adaptation. This technology is critical for enhancing:
- Pedestrian safety both indoors and outdoors.
- Providing clear information about the user's current location.
- Supporting effecture navigation toward the desire destination.

## Research Motivation

- Limitations of Current Tools:  
While traditional mobility aids such as white canes and guide dogs provide essential support, they have limitations in detecting distant or overhead obstacles and identifying clear paths in complex environments
- Indoor vs Outdoor Gap:  
GPS works well for outside navigation, but it is unreliable indoors. Additionally, there is a lack of seamless mobility support due to the limitations of current indoor navigation solutions.
- Orientation Challenges:  
Many existing solutions fail to provide clear and accessible real-time information about a user's exact location, leading to disorientation and reduced confidence in mobility.

## Related Work

- [Real-Time Wayfinding Assistant for Blind and Low-Vision Users](https://arxiv.org/html/2504.20976v1)
- [Visual Impairment Spatial Awareness System for Indoor Navigation and Daily Activities](https://pmc.ncbi.nlm.nih.gov/articles/PMC11766877/)  
- [Technological Advancements in Human Navigation for the Visually Impaired: A Systematic Revies](https://www.mdpi.com/1424-8220/25/7/2213)

## Problems faced in Visually Impaired Navigator and Solutions

- Accurate Localisation.  
Clock Hour Hand balances accuracy and user intuitiveness, making it a practical solution for real-world applications.  
![Diagram of navigation scheme to define movement direction.](docs/Imaging/imgs/Clock_Hour_Hand.png) 
- Data processing.    
Depth-First Search algorithm offers several advantages, includign lightweight computation, real-time performance, and greater interpretability.  
![Flow diagram for identifying the free path.](docs/Imaging/imgs/Depth-First_Search.png)
- Cost barriers.  

## Related Concepts

- Navigation
- Object Detection and Recognition
- Depth Profiling

## Expected Results

- Localization
  - Pinpoint the user's indoor and outdoor position with minimal error.
- Obstacle Detection and Avoidance
  - Clear auditory feedback for safe path navigation.
- Path Planning and Guidance
  - Generate and continuously update the safest route in both indoor and outdoor.
