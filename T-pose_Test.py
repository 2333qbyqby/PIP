import time
import sys
import pathlib
import keyboard
# 添加项目根目录到Python路径
project_root = pathlib.Path(__file__).parent
sys.path.append(str(project_root))
# 导入配置
from config import paths
from articulate.utils.rbdl import *
from articulate.utils.bullet import *
import numpy as np
import pybullet as p
from utils import *
def test_t_pose():
    """
    测试T-pose姿势显示
    """
    mu = 0.6
    supp_poly_size = 0.2
    model = RBDLModel(paths.physics_model_file, update_kinematics_by_hand=True)
    params = read_debug_param_values_from_json(paths.physics_parameter_file)
    friction_constraint_matrix = np.array([[np.sqrt(2), -mu, 0],
                                                [-np.sqrt(2), -mu, 0],
                                                [0, -mu, np.sqrt(2)],
                                                [0, -mu, -np.sqrt(2)]])
    support_polygon = np.array([[-supp_poly_size / 2, 0, -supp_poly_size / 2],
                                     [supp_poly_size / 2, 0, -supp_poly_size / 2],
                                     [-supp_poly_size / 2, 0, supp_poly_size / 2],
                                     [supp_poly_size / 2, 0, supp_poly_size / 2]])

    p.connect(p.GUI)
    p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1)
    id_robot = p.loadURDF(paths.physics_model_file, [0, 0, 0], useFixedBase=False,
                                   flags=p.URDF_MERGE_FIXED_LINKS)
    change_color(id_robot, [198 / 255, 238 / 255, 0, 1.0])
    p.loadURDF(paths.plane_file, [0, -0.881, 0.0], [-0.7071068, 0, 0, 0.7071068])
    load_debug_params_into_bullet_from_json(paths.physics_parameter_file)
    # 设置T-pose（所有关节角度为0）
    # 构造75维的配置向量q，其中前3个是根位置，接下来3个是根旋转，其余是关节角度
    q = np.zeros(75)
    q[3] = 0  # 绕z轴的旋转角度设为0（保持直立）
    q[:3] = np.array([0, 1, 0])
    set_pose(id_robot, q)
    # 应用姿势到机器人
    print("T-pose已设置完成")
    print("按Ctrl+C退出...")

    # 保持程序运行，直到用户中断
    try:
        while True:
            camera_info = p.getDebugVisualizerCamera()
            current_yaw, current_pitch, current_distance = camera_info[5][0], camera_info[4][1], camera_info[11][2]
            if keyboard.is_pressed('a'):
                p.resetDebugVisualizerCamera(
                    cameraDistance=2.5,
                    cameraYaw=current_yaw + 1,
                    cameraPitch=current_pitch,
                    cameraTargetPosition=[0, 0, 0]
                )
            #p.stepSimulation()
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        print("\n程序退出")
    finally:
        p.disconnect()





if __name__ == "__main__":
    test_t_pose()