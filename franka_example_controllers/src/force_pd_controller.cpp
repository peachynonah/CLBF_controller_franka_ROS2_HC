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

#include <franka_example_controllers/force_pd_controller.hpp>
#include <franka_example_controllers/robot_utils.hpp>
#include <franka_example_controllers/default_robot_behavior_utils.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <string>
#include <iostream>

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <unsupported/Eigen/EulerAngles>
#include <kdl/jntarray.hpp> //초기 조인트값 정의를 위한
#include <kdl/frames.hpp>
#include <kdl/jntspaceinertiamatrix.hpp>

#include <geometry_msgs/msg/point_stamped.hpp>  // Publishers for bag recording
#include <std_msgs/msg/float64_multi_array.hpp>  // Publishers for bag recording

#include <rclcpp/rclcpp.hpp>


namespace franka_example_controllers {



//////////////////////////////////////////////////////////////////////////////
/////############### 1. ROS2 ControllerInterface functions ##############/////
/////////////////////////////////////////////////////////////////////////////


controller_interface::InterfaceConfiguration
ForcePDController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
ForcePDController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // config.names = franka_cartesian_pose_->get_state_interface_names();

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

controller_interface::return_type ForcePDController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {
  updateJointStates();
  elapsed_time_ = elapsed_time_ + period.seconds();

  KDL::JntArray q_kdl(num_joints);
  KDL::JntArray dq_kdl(num_joints);


  const double kAlpha = 0.99;
  dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * dq_;

  for (int i = 0; i < num_joints; ++i) {
    q_kdl(i) = q_(i);
    dq_kdl(i) = dq_filtered_(i);    
    q_goal_(i) = initial_q_(i); //##q_goal 확인하기
    q_total_(i) = q_(i);//##q_확인하기
    dq_total_(i) = dq_filtered_(i);    //##dq_ 확인하기    
  }

  getCartesianState(q_kdl);
  getCartesianGoal();
  getCartesianError();

  force_calculated = k_gains_.cwiseProduct(cart_pos_err) + d_gains_.cwiseProduct(cart_vel_err);


  Vector7d tau_d_calculated = jac_ana_eigen.transpose() * force_calculated;
  Vector7d tau_total = tau_d_calculated;
  
  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_total(i));
  }


  // debugging output, debuging count: 500/1000 = printing rate
    static std::size_t dbg_cnt = 0;
  if (++dbg_cnt % 500 == 0) {
    std::cout << "[DBG] geometric jacobian = [" << jac_eigen
              << "]" << std::endl;
    std::cout << "[DBG] analytic jacobian = [" << jac_ana_eigen
              << "]" << std::endl;
    std::cout << "[DBG] tau_total = [" << tau_total
              << "]" << std::endl;
    std::cout << "[DBG] cartesian pose error = [" << cart_pos_err.transpose()
              << "]" << std::endl;
    std::cout << " " << std::endl;
  }
  
  //Publish
  publish_msg(tau_total, q_total_, q_goal_, dq_total_);
    return controller_interface::return_type::OK;
}


CallbackReturn ForcePDController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "");
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});

  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn ForcePDController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  if (k_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains parameter not set");
    return CallbackReturn::FAILURE;
  }
  if (k_gains.size() != static_cast<uint>(crt_dim)) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains should be of size %d but is of size %ld",
                 crt_dim, k_gains.size());
    return CallbackReturn::FAILURE;
  }
  if (d_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_gains parameter not set");
    return CallbackReturn::FAILURE;
  }
  if (d_gains.size() != static_cast<uint>(crt_dim)) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_gains should be of size %d but is of size %ld",
                 crt_dim, d_gains.size());
    return CallbackReturn::FAILURE;
  }
  for (int i = 0; i < crt_dim; ++i) {
    d_gains_(i) = d_gains.at(i);
    k_gains_(i) = k_gains.at(i);
  }
  dq_filtered_.setZero();
  // initial_q_ = q_;

  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
  parameters_client->wait_for_service();

  auto future = parameters_client->get_parameters({"robot_description"});
  auto result = future.get();
  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
  } else {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
  }
  //KDL 기반 로봇 동역학 모델 객체 초기화
  //(1)기본 체인 생성
  kdl_model_param_ = std::make_unique<KDLModelParam>(robot_description_, "fr3_link0", "fr3_link7");
  if (!kdl_model_param_->isValid()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to initialize KDL model param.");
    return CallbackReturn::FAILURE;
  }
  // 이미 robot_description_ 을 갖고 있으니 그대로 파싱
  KDL::Tree tree;
  if (!kdl_parser::treeFromString(robot_description_, tree)) {
    RCLCPP_ERROR(get_node()->get_logger(), "KDL tree parse failed");
    return CallbackReturn::FAILURE;
  }

  arm_id_ = robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());
  // --- Publishers ---
  ee_position_pub_ = get_node()->create_publisher<geometry_msgs::msg::PointStamped>(
    "ee_position", rclcpp::SystemDefaultsQoS());
  ee_orientation_pub_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
    "ee_orientation_rpy", rclcpp::SystemDefaultsQoS());
  cart_pos_err_pub_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
    "cart_pos_err", rclcpp::SystemDefaultsQoS());
  tau_total_pub_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
    "tau_total", rclcpp::SystemDefaultsQoS());
  cart_goal_pub_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
    "cart_goal", rclcpp::SystemDefaultsQoS());
//##q, dq에 대해 출력
  q_total_pub_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
    "q_total", rclcpp::SystemDefaultsQoS());
  q_goal_pub_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
    "q_goal", rclcpp::SystemDefaultsQoS());    
  dq_total_pub_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
    "dq_total", rclcpp::SystemDefaultsQoS());    
  return CallbackReturn::SUCCESS;
}

CallbackReturn ForcePDController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  updateJointStates();
  dq_filtered_.setZero();
  initial_q_ = q_;
  initializeCartesianState(initial_q_);
  elapsed_time_ = 0.0; 
  return CallbackReturn::SUCCESS;
}



/////////////////////////////////////////////////////////////////////////////////
/////############### 2. additional function part ###########################/////
/////////////////////////////////////////////////////////////////////////////////



// 1. Joint States and joint dynamics model
void ForcePDController::updateJointStates() {
  for (auto i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}


 // 2. Jacobian terms
 Eigen::Matrix<double, 6,7> ForcePDController::getCrtAnalyticJacobian(KDL::JntArray q_in_form_of_kdl){
  // geometric 자코비안 계산 (6x7 행렬)
    KDL::Jacobian jac_kdl(num_joints);
    kdl_model_param_->computeJacobian(q_in_form_of_kdl, jac_kdl);

    // data type transfer of jacobian: KDL::Jacobian to Eigen::Matrix 
      for (unsigned int i = 0; i < 6; ++i) {
          for (unsigned int j = 0; j < num_joints; ++j) {
            jac_eigen(i, j) = jac_kdl(i, j);
          }
      }

    //geometric <> analytic jacobian conversion: Euler ZYX angle (fixed RPY method)
    spsi = std::sin(euler_psi),  cpsi = std::cos(euler_psi);
    stheta = std::sin(euler_theta), ctheta = std::cos(euler_theta);

    T_ZYX_inverse << (cpsi*stheta)/(ctheta), (stheta*spsi)/(ctheta), 1,
                     -spsi, cpsi, 0 , 
                     cpsi/ctheta, spsi/ctheta, 0;
    
    T_A_ZYX_inverse.bottomRightCorner<3,3>() = T_ZYX_inverse;
    jac_ana_eigen = T_A_ZYX_inverse * jac_eigen;
    
    return jac_ana_eigen;
 }


// 3. Functions of Cartesian space pose
void ForcePDController::initializeCartesianState(Vector7d initial_q_){
   KDL::JntArray q_initial_kdl(num_joints);

    for (int i = 0; i < num_joints; ++i) {
      q_initial_kdl(i) = initial_q_(i);
    }

  kdl_model_param_->computeForwardKinematics(q_initial_kdl, ee_initial_frame_);
  
  //initial position
  cart_initial(0) = ee_initial_frame_.p.x();
  cart_initial(1) = ee_initial_frame_.p.y();
  cart_initial(2) = ee_initial_frame_.p.z();

  //initial orientation
  ee_initial_frame_.M.GetEulerZYX(initial_psi, initial_theta, initial_phi);
  initial_psi   = wrapToPi(initial_psi);
  initial_theta = wrapToPi(initial_theta);
  initial_phi   = wrapToPi(initial_phi);
  
  cart_initial(3) = initial_psi;
  cart_initial(4) = initial_theta;
  cart_initial(5) = initial_phi;  

}

void ForcePDController::getCartesianState(const KDL::JntArray &q_kdl){
 
  kdl_model_param_->computeForwardKinematics(q_kdl, ee_frame_);

  //current position
  cart_pos_current(0) = ee_frame_.p.x();
  cart_pos_current(1) = ee_frame_.p.y();
  cart_pos_current(2) = ee_frame_.p.z();

  //current orientation
  ee_frame_.M.GetEulerZYX(euler_psi, euler_theta, euler_phi);
  euler_psi   = wrapToPi(euler_psi);
  euler_theta = wrapToPi(euler_theta);
  euler_phi   = wrapToPi(euler_phi);

  cart_pos_current(3) = euler_psi;
  cart_pos_current(4) = euler_theta;
  cart_pos_current(5) = euler_phi;

  //current catesian velocity
  jac_ana_eigen = getCrtAnalyticJacobian(q_kdl);
  cart_vel_current = jac_ana_eigen * dq_;
}

void ForcePDController::getCartesianGoal(){
  
  //initialize. maybe it's calculated for every update rate, so this initializing should move to on_activate, or deleted. 
  // if delete this part, robot pose explodes(...)
  cart_pos_goal = cart_initial;

  //references.
  double radius = 0.3;
  double angle = M_PI / 4 * (1 - std::cos((M_PI / 5.0) * elapsed_time_));

  double delta_x = radius * std::sin(angle);
  double delta_z = radius * (std::cos(angle) - 1);
  double delta_phi = std::sin(angle);
  double delta_psi = std::sin(angle);

  cart_pos_goal(0) -= delta_x;
  cart_pos_goal(2) -= delta_z;
  cart_pos_goal(5) += delta_phi;


  //velocity goal update: should be updated based on reference velocity. 
  cart_vel_goal = Vector6d::Zero();
}

void ForcePDController::getCartesianError(){
  
  //pose error update. maybe it's calculated for every update rate, so this initializing should move to on_activate, or deleted. 
  cart_pos_err = cart_pos_goal - cart_pos_current;

  //orientation error wrapping
  cart_pos_err(3) = wrapToPi(cart_pos_err(3));
  cart_pos_err(4) = wrapToPi(cart_pos_err(4));
  cart_pos_err(5) = wrapToPi(cart_pos_err(5));

  //velocity error update
  cart_vel_err = cart_vel_goal - cart_vel_current;

  // No idea of "should euler velocity also be wrapped?"
}



// 5. Functions of mathmatical calculation
int ForcePDController::matrixRank(const Eigen::MatrixXd & M, double tol)
{
  Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
  if (tol < 0) tol = lu.threshold();
  return lu.rank();
}

double ForcePDController::wrapToPi(double rad)
{
 double two_pi = 2.0 * M_PI;
  while (rad >  M_PI)  rad -= two_pi;
  while (rad <= -M_PI) rad += two_pi;
  return rad;
}



// 6. Functions of publishing datas for graph
void ForcePDController::publish_msg(Vector7d tau_total, Vector7d q_total_, Vector7d q_goal_, Vector7d dq_total_){

  // === 1) ee_frame_.p 퍼블리시 ===
  {
    geometry_msgs::msg::PointStamped msg;
    msg.header.stamp = get_node()->now();
    msg.header.frame_id = "base_link";  // 필요에 따라 변경
    msg.point.x = ee_frame_.p.x();
    msg.point.y = ee_frame_.p.y();
    msg.point.z = ee_frame_.p.z();
    ee_position_pub_->publish(msg);
  }

  // === 2) EE orientation (roll-pitch-yaw, ZYX) publish ===
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.layout.dim.resize(1);
    msg.layout.dim[0].label  = "rpy";
    msg.layout.dim[0].size   = 3;
    msg.layout.dim[0].stride = 3;
    msg.data = {euler_psi,   // Z-axis rotation (roll)
                euler_theta, // Y-axis rotation (pitch)
                euler_phi};  // X-axis rotation (yaw)
    ee_orientation_pub_->publish(msg);
  }

  // === 3) cart_pos_err 퍼블리시 ===
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.layout.dim.resize(1);
    msg.layout.dim[0].label = "error6";
    msg.layout.dim[0].size = 6;
    msg.layout.dim[0].stride = 6;
    msg.data = std::vector<double>(cart_pos_err.data(), cart_pos_err.data() + 6);
    cart_pos_err_pub_->publish(msg);
  }

  // === 4) tau_total 퍼블리시 ===
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.layout.dim.resize(1);
    msg.layout.dim[0].label = "tau7";
    msg.layout.dim[0].size = 7;
    msg.layout.dim[0].stride = 7;
    msg.data = std::vector<double>(tau_total.data(), tau_total.data() + 7);
    tau_total_pub_->publish(msg);
  }

    // === 5) cart_goal 퍼블리시 ===
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.layout.dim.resize(1);
    msg.layout.dim[0].label = "cart_goal";
    msg.layout.dim[0].size = 6;
    msg.layout.dim[0].stride = 6;
    msg.data = std::vector<double>(cart_pos_goal.data(), cart_pos_goal.data() + 6);
    cart_goal_pub_->publish(msg);
  }

    // === 6) q_total 퍼블리시 ===
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.layout.dim.resize(1);
    msg.layout.dim[0].label = "q_total";
    msg.layout.dim[0].size = 7;
    msg.layout.dim[0].stride = 7;
    msg.data = std::vector<double>(q_total_.data(), q_total_.data() + 7);
    q_total_pub_->publish(msg);
  }    

      // === 7) q_goal 퍼블리시 ===
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.layout.dim.resize(1);
    msg.layout.dim[0].label = "q_goal";
    msg.layout.dim[0].size = 7;
    msg.layout.dim[0].stride = 7;
    msg.data = std::vector<double>(q_goal_.data(), q_goal_.data() + 7);
    q_goal_pub_->publish(msg);
  }

      // === 8) dq_total 퍼블리시 ===
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.layout.dim.resize(1);
    msg.layout.dim[0].label = "dq_total";
    msg.layout.dim[0].size = 7;
    msg.layout.dim[0].stride = 7;
    msg.data = std::vector<double>(dq_total_.data(), dq_total_.data() + 7);
    dq_total_pub_->publish(msg);
  }

  
}

}  // namespace franka_example_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::ForcePDController,
                       controller_interface::ControllerInterface)