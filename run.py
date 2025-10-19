
from services.scenarios import simple_pn_intercept
import time

def main():
    world = simple_pn_intercept()
    dt = 0.05
    for i in range(int(60.0 / dt)):
        world.step(dt)
        if i % 10 == 0:
            missile = world.entities[0]
            target = world.entities[1]
            print(f"t={world.t:.2f}   missile_r={missile.state.r}   target_r={target.state.r}")
        time.sleep(0.0)  # set to nonzero if you want to slow down

if __name__ == "__main__":
    main()
