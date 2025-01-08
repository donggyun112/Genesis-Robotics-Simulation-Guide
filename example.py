import genesis as gs
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

class FrankaEnv:
    def __init__(self):
        # 시뮬레이션 설정
        self.control_freq = 50
        self.dt = 1.0 / self.control_freq
        
        # 씬 생성
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, -3.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=60
            ),
            show_viewer=True,
            rigid_options=gs.options.RigidOptions(
                dt=0.01,
                gravity=(0.0, 0.0, -10.0),
            ),
        )
        
        # 로봇 및 환경 구성
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
        )
        self.target = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04), 
                pos=(0.5, 0.0, 0.02)
            )
        )
        
        # 제어 설정
        self.arm_joints = np.arange(7)
        self.gripper_joints = np.arange(7, 9)
        
        # 씬 빌드
        self.scene.build()
        
        # PD 제어기 설정
        self.franka.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
        )
        self.franka.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])
        )
        
        self.last_action = np.zeros(9)

    def reset(self):
        # 초기 자세로 리셋
        initial_pos = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.5, 0.0, 0.04, 0.04])
        self.franka.set_dofs_position(initial_pos)
        
        # 물체 위치 랜덤 설정 - set_position -> set_pos
        target_pos = np.array([
            np.random.uniform(0.3, 0.7),
            np.random.uniform(-0.3, 0.3),
            0.02
        ])
        self.target.set_pos(target_pos)  # set_position() -> set_pos()
        
        return self.get_observation()


    def get_observation(self):
        # 상태 관측
        joint_pos = self.franka.get_dofs_position().cpu().numpy()  # cpu로 이동 후 numpy로 변환
        joint_vel = self.franka.get_dofs_velocity().cpu().numpy()
        ee_link = self.franka.get_link('hand')
        
        ee_pos = ee_link.get_pos().cpu().numpy()
        target_pos = self.target.get_pos().cpu().numpy()
        
        return np.concatenate([
            joint_pos,
            joint_vel,
            ee_pos,
            target_pos
        ])




    def compute_reward(self):
        # 엔드이펙터와 물체 사이의 거리
        ee_link = self.franka.get_link('hand')
        ee_pos = ee_link.get_pos().cpu().numpy()
        target_pos = self.target.get_pos().cpu().numpy()
        distance = np.linalg.norm(ee_pos - target_pos)
        
        # 그리퍼가 물체를 잡았는지 확인
        gripper_pos = self.franka.get_dofs_position().cpu().numpy()[7:9]
        has_grasped = np.all(gripper_pos < 0.02)
        
        # 보상 계산
        position_reward = -distance
        grasp_reward = 10.0 if has_grasped else 0.0
        stability_reward = -0.1 * np.mean(np.abs(self.last_action))
        
        return position_reward + grasp_reward + stability_reward

    def step(self, action):
        self.last_action = action
        
        ee_pos = self.franka.get_link('hand').get_pos().cpu().numpy()
        target_pos = self.target.get_pos().cpu().numpy()
        
        # 바닥과의 충돌 체크 (z축이 너무 낮은지)
        if ee_pos[2] < 0.01:  # 더 낮은 값으로 설정
            return self.get_observation(), -10.0, True, {}
        
        # 액션 실행
        self.franka.control_dofs_position(action[:7], self.arm_joints)
        self.franka.control_dofs_position(action[7:], self.gripper_joints)
        
        self.scene.step()
        
        obs = self.get_observation()
        reward = self.compute_reward()
        
        # 성공 조건 수정
        gripper_pos = self.franka.get_dofs_position().cpu().numpy()[7:9]
        distance = np.linalg.norm(ee_pos - target_pos)
        
        # 물체를 성공적으로 잡았을 때만 종료
        if distance < 0.05 and np.all(gripper_pos < 0.02):
            print("Successfully grasped the object!")
            return obs, reward + 50.0, True, {}
            
        return obs, reward, False, {}

# PPO 정책 네트워크
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim * 2)  # mean과 std 출력
        )
        
    def forward(self, x):
        x = self.actor(x)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        dist = Normal(mean, std)
        return dist

def run_sim(env, enable_vis):
    obs_dim = len(env.get_observation())
    act_dim = 9
    policy = Policy(obs_dim, act_dim)
    
    # 학습 루프
    for episode in range(1000):
        obs = env.reset()
        episode_reward = 0
        
        # 1000 -> 300으로 감소
        for step in range(300):  # 한 번의 시도에 적합한 길이로 조정
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs)
                dist = policy(obs_tensor)
                action = dist.sample().numpy()
            
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                print(f"Episode {episode}, Steps: {step}, Reward: {episode_reward}")
                break
        
        print(f"Episode {episode}, Reward: {episode_reward}")

if __name__ == "__main__":
    # Genesis 초기화
    gs.init(backend=gs.metal)
    
    # 환경 생성
    env = FrankaEnv()
    
    # 별도 스레드에서 시뮬레이션 실행
    gs.tools.run_in_another_thread(
        fn=run_sim, 
        args=(env, True)
    )
    
    # 뷰어 시작
    env.scene.viewer.start()