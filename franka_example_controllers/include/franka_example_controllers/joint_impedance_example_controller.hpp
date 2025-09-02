// Copyright (c) 2023 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>

#include <Eigen/Eigen>
#include <Eigen/Dense>
// #include <Eigen/Core>
// #include <Eigen/Geometry>

#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include "hardware_interface/loaned_state_interface.hpp"
#include <tf2_eigen_kdl/tf2_eigen_kdl.hpp>

//added
#include <franka_example_controllers/robot_utils.hpp>
#include <franka_semantic_components/franka_cartesian_pose_interface.hpp>

#include <franka_example_controllers/kdl_model_param.hpp>
// joint_impedance_example_controller.hpp (또는 .cpp 상단)
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>  // FK-solver
#include <kdl/tree.hpp>                        // KDL::Tree
#include <kdl_parser/kdl_parser.hpp>           // kdl_parser::treeFromString
// #include <kdl/frames.hpp>

//added


#include <franka/robot_state.h>
#include <controller_interface/controller_interface.hpp>
#include <franka_msgs/msg/franka_robot_state.hpp>
#include "franka_semantic_components/franka_robot_model.hpp"
#include "franka_semantic_components/franka_robot_state.hpp"





using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

/**
 * The joint impedance example controller moves joint 4 and 5 in a very compliant periodic movement.
 */
class JointImpedanceExampleController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
    // added 
  //CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;
  // added

 private:
  // //added
  // std::unique_ptr<franka_semantic_components::FrankaCartesianPoseInterface> franka_cartesian_pose_;
  // void printCartesianStates();
  // //added

  // Eigen::Quaterniond orientation_;
  // // Eigen::Vector3d position_;
  // const bool k_elbow_activated_{false};
  // bool initialization_flag_{true};

  std::string arm_id_;
  std::string robot_description_;
  const int num_joints = 7;
  Vector7d q_;
  Vector7d initial_q_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector7d k_gains_;
  Vector7d d_gains_;
  double elapsed_time_{0.0};
  void updateJointStates();
  void getKDLmodel();
   // KDL 모델 파라미터 계산 객체
  std::unique_ptr<KDLModelParam> kdl_model_param_; // 추가된 부분
  KDL::Chain kdl_chain_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_pos_solver_;
  KDL::Frame ee_frame_;        // 결과 저장용 (위치+자세)
  Eigen::Quaterniond ee_orientation_;
};

}  // namespace franka_example_controllers
