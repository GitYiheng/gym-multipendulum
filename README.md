# gym-multipendulum

It works for Ubuntu 16.04. But I have not tested it with Windows or Mac.

Issue or contact me if you find any problem: gityiheng@gmail.com

### Prerequisite

```bash
apt-get install -y python3-pip python3-numpy python3-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```

### Install OpenAI Gym

```bash
git clone https://github.com/openai/gym.git
cd gym
pip3 install -e .
cd ..
```

### Install gym-multipendulum

```bash
git clone https://github.com/GitYiheng/gym-multipendulum.git
cd ./gym-multipendulum
pip3 install -e .
cd ..
```

### Test gym-multipendulum

```python
import gym
import gym_multipendulum

env = gym.make('multipendulum-v0')
env.reset()
done = False
reward = 0

while not done:
	action = env.sample_action()
	observation, reward, done, _ = env.step(action)
	env.render()
```
