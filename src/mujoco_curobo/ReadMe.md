```mermaid
graph TD;
    subgraph System
        A[meshes, urdf] -->|Input| B[Isaac SIM]
        A --> C[Initial World xml]
        B --> D[Collision spheres.yml]
        C --> E[Mujoco]
        D --> F[Curobo]
        E -->|World Config| G[World Config]
        G --> F
        F -->|Motion Plan| E
    end


graph TD;
    A[Home] -->|plan_single| B[Pre-pick]
    B -->|plan_single<br>Open gripper| C[Next object location]
    C -->|plan_single<br>Attach object to robot<br>Close gripper| D[Post-pick]
    D -->|plan_single| E[Pre-place]
    E -->|plan_single<br>Detach object from robot<br>Open gripper| F[Place]
    F -->|plan_single_j| A


```


