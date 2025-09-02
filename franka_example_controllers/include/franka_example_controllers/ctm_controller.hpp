#pragma once

#include <string>

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <unsupported/Eigen/EulerAngles>
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
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>  // FK-solver
#include <kdl/tree.hpp>                        // KDL::Tree
#include <kdl_parser/kdl_parser.hpp>           // kdl_parser::treeFromString
#include <kdl/chainjnttojacdotsolver.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/frames.hpp>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <Eigen/Dense> 
#include <kdl/jntspaceinertiamatrix.hpp>


using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

/**
 * The joint impedance example controller moves joint 4 and 5 in a very compliant periodic movement.
 */
class CTMController : public controller_interface::ControllerInterface {
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
  using Vector6d = Eigen::Matrix<double, 6, 1>;

  // Publishers for bag recording
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr ee_position_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr ee_orientation_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cart_pos_err_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr tau_total_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cart_goal_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr q_total_pub_;   //##q, dq추가
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr q_goal_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr dq_total_pub_;

 private:
  std::string arm_id_;
  std::string robot_description_;
  const int num_joints = 7;
  const int crt_dim = 6;

  Vector7d initial_q_;
  Vector7d dq_filtered_;
  Vector6d k_gains_;
  Vector6d d_gains_;  
  double elapsed_time_{0.0};


  //update
  Vector7d q_goal_, q_total_, dq_total_;
  Vector7d tau_total;

  //updateJointStates
  void updateJointStates();
  Vector7d q_, dq_;


  //getKDLmodel
  void getKDLmodel(const KDL::JntArray &q_kdl, const KDL::JntArray &dq_kdl);
  std::unique_ptr<KDLModelParam> kdl_model_param_;
  KDL::Chain kdl_chain_;
  KDL::JntArray gravity_; 
  KDL::JntArray coriolis_;
  KDL::JntSpaceInertiaMatrix mass_matrix_; //null space에서 사용할 mass matrix 저장할 변수
  Eigen::Matrix<double, 7, 7> M_eig;
  Eigen::Matrix<double, 7, 1> C_eig, G_eig;


 //getCrtAnalyticJacobian
 Eigen::Matrix<double, 6, 7> getCrtAnalyticJacobian(KDL::JntArray q_in_form_of_kdl);
 double spsi, cpsi, stheta, ctheta;
 Eigen::Matrix3d T_ZYX, T_ZYX_inverse;
 Eigen::Matrix<double, 6, 6> T_A_ZYX = Eigen::Matrix<double, 6, 6>::Identity();
 Eigen::Matrix<double, 6, 6> T_A_ZYX_inverse = Eigen::Matrix<double, 6, 6>::Identity();
 Eigen::Matrix<double, 6, 7> jac_eigen, jac_ana_eigen, jac_ana_eigen_prev, jac_ana_eigen_dot;


 //getAnalyticJacobianDotQdot
 Eigen::Matrix<double,6,1>getAnalyticJacobianDotQdot(const KDL::JntArray& q,const KDL::JntArray& qdot);
 
 //수치미분을 위해 추가
//  Eigen::Matrix<double,6,1> jdqd_ana_eigen;
//  Eigen::Matrix<double,6,7> jd_ana_eigen;
//  filters::JacobianLPF jdot_filter_{0.0, 20.0, 7};
//  double dt {0.0};

 //initializeCartesianState
 void initializeCartesianState(Vector7d initial_q_);
 KDL::Frame ee_initial_frame_;
 Vector6d cart_initial;
 double initial_phi {0.0};
 double initial_theta {0.0};
 double initial_psi {0.0};    


 //getCartesianState
 void getCartesianState(const KDL::JntArray &q_kdl);
 std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
 KDL::Frame ee_frame_;
 Eigen::Quaterniond ee_ori_quat_;
 Eigen::Matrix3d ee_ori_rot_;
 Eigen::EulerAngles<double, Eigen::EulerSystemZYX> ee_ori_eulerZYX_;
 double euler_phi {0.0};
 double euler_theta {0.0};  
 double euler_psi {0.0};  
 Vector6d cart_pos_current, cart_vel_current;


 //getCartesianGoal
 void getCartesianGoal();
 Vector6d cart_pos_goal, cart_vel_goal; 


 //getCartesianError
 void getCartesianError(); 
 Vector6d cart_pos_err, cart_vel_err;
 Vector7d gravity_matrix_;

 //CTMinput
 void getCTMinput(); 
 Vector6d force_error, force_calculated;
 Vector7d tau_d_calculated;
 
 //##null space controller에서 2개 동시 출력
    using NullspaceResult = std::pair<
      Eigen::Matrix<double, 7, 1>,   // tau_n
      Eigen::Matrix<double, 7, 6>    // Jbar
  >;

  // Eigen::Matrix<double, 7, 1> Nullspace_controller(
    NullspaceResult Nullspace_controller(
      const Eigen::Matrix<double, 6, 7>& jac_ana,
      const Eigen::Matrix<double, 7, 1>& K0);

//##getCTMingredient에서 2개 동시 출력
    using CTMingredientResult = std::pair<
      Eigen::Matrix<double, 6, 6>,   // M_x
      Eigen::Matrix<double, 6, 1>    // C_x
  >;

  //getCTMingredient
  CTMingredientResult getCTMingredient(
    const Eigen::Matrix<double, 6, 7>& jac_ana,
    /*const Eigen::Matrix<double, 6, 1>& jdqd_ana_eigen,*/
    const Eigen::Matrix<double, 7, 7>& M_eig,
    const Eigen::Matrix<double, 7, 1>& C_eig);

  Eigen::Matrix<double, 7, 7> M_eig_inv;
  Eigen::Matrix<double, 6, 6> lambda_inv, lambda, M_x;
  Eigen::Matrix<double, 7, 6> Jbar;
  Eigen::Matrix<double, 6, 1> C_x;


  int matrixRank(const Eigen::MatrixXd & M_eig, double tol = -1.0); //rank 확인용 함수
  double wrapToPi(double rad);
  void publish_msg(Vector7d tau_total, Vector7d q_total_, Vector7d q_goal_, Vector7d dq_total_);

  
};
}  // namespace franka_example_controllers