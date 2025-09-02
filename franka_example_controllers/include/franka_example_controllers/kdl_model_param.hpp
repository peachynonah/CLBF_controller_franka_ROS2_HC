#pragma once

#include <kdl/chain.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/jntspaceinertiamatrix.hpp>
#include <kdl/chainjnttojacsolver.hpp> // 자코비안 계산 추가
#include <kdl/chainjnttojacdotsolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp> // FK 계산 추가
#include <kdl/frames.hpp> // 프레임 계산 추가
#include <memory>
#include <string>

namespace franka_example_controllers {

class KDLModelParam {
public:
  KDLModelParam(const std::string& urdf_string, const std::string& root, const std::string& tip);
  bool isValid() const;

  bool computeDynamics(const KDL::JntArray& q,const KDL::JntArray& dq,KDL::JntSpaceInertiaMatrix& mass,KDL::JntArray& coriolis, KDL::JntArray& gravity);
  bool computeJacobian(const KDL::JntArray& q, KDL::Jacobian& jacobian);  // 자코비안 계산 (6x7 행렬)
  bool computeForwardKinematics(const KDL::JntArray& q, KDL::Frame& end_effector_pose); // FK
  bool computeJacobianQDot(const KDL::JntArray& q,const KDL::JntArray& dq,KDL::Twist& jdot_qdot); // 자코비안 미분 * dq
  bool computeJacobianDot(const KDL::JntArray& q,const KDL::JntArray& dq,KDL::Jacobian& Jdot); //자코비안 미분

private:
  KDL::Chain chain_;
  std::unique_ptr<KDL::ChainDynParam> dyn_param_;
  std::unique_ptr<KDL::ChainJntToJacSolver> jac_solver_; // 자코비안 계산
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_; // FK
  std::unique_ptr<KDL::ChainJntToJacDotSolver> jdot_solver_; // 자코비안 미분*dq, 자코비안 미분 계산
  bool valid_;
};

} // namespace franka_example_controllers
