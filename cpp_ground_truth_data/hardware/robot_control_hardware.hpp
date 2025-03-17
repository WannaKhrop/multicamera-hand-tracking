#pragma once

#include <franka_proxy_client/franka_remote_interface.hpp>
#include <Eigen/Geometry>

#include "robot_control.hpp"

class camera;

class robot_control_hardware : public robot_control
{

public:

	/*
	* @param proxy_ip = ip of franka proxy server
	*/
	robot_control_hardware(const std::string& proxy_ip);

	robot_control_hardware(const robot_control_hardware& hardware);

	/*
	* moves to given configuration
	* @param destination joint configuration of destination
	*/
	virtual void move_to_configuration(const franka_proxy::robot_config_7dof& destination) override;

	
	/*
	* returns current robot config
	* @return current robot config
	*/
	virtual franka_proxy::robot_config_7dof get_current_config() override;


	virtual void update() override;

	virtual void get_robot_coordinates_for_hand() override;


private:
	void move_without_errors_to_configuration(const franka_proxy::robot_config_7dof& destination);
	static Eigen::Vector3d get_nsa_position(const franka_proxy::robot_config_7dof& current_pos);
	static Eigen::Affine3d get_nsa_frame(const franka_proxy::robot_config_7dof& current_pos);

	


	franka_proxy::franka_remote_interface controller_;
	std::thread update_thread_;
};

