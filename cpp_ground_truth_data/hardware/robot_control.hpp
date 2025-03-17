#pragma once

#include <franka_proxy_share/franka_proxy_util.hpp>
#include <franka_proxy_client/franka_remote_interface.hpp>
#include <franka_control/franka_controller_emulated.hpp>
#include <Eigen/Geometry>


class camera;

/*
* updater class for franka proxy update
*/
class robot_updater
{
public:
	robot_updater();

	virtual void update() = 0;

	void start_thread();
	void stop_thread();

	~robot_updater();
private:
	static constexpr int waiting_time = 16;
	void update_loop();
	bool is_running_;
	std::thread thread_;
};

class robot_control : public robot_updater
{

public:

	/*
	* @param proxy_ip = ip of franka proxy server
	*/
	robot_control(const std::string& proxy_ip);

	/*
	* converts @param in to franka_control::robot_config_7dof format
	*/
	static franka_control::robot_config_7dof change_pose_to_franka_control(const franka_proxy::robot_config_7dof& in);

	/*
	* converts @param in to franka_proxy::robot_config_7dof format
	*/
	static franka_proxy::robot_config_7dof change_pose_to_franka_proxy(const franka_control::robot_config_7dof& in);

	/*
	* moves to given configuration
	* @param destination joint configuration of destination
	*/
	virtual void move_to_configuration(const franka_proxy::robot_config_7dof& destination) = 0;


	/*
	* returns current robot config
	* @return current robot config
	*/
	virtual franka_proxy::robot_config_7dof get_current_config() = 0;

	virtual void get_robot_coordinates_for_hand() = 0;


	static franka_control::robot_config_7dof calculate_joints_from_cartesian(const Eigen::Affine3d& nsa);

	virtual void update() = 0;
protected:
	const std::string ip;
};

