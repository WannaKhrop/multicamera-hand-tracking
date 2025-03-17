#include "robot_control_hardware.hpp"

#include <iostream>
#include <fstream>
#include <franka_proxy_share/franka_proxy_util.hpp>


robot_control_hardware::robot_control_hardware(const std::string& proxy_ip) :
	robot_control(proxy_ip),
	controller_(ip)
{
	controller_.set_speed_factor(0.15);
	start_thread();
}
robot_control_hardware::robot_control_hardware(const robot_control_hardware& hardware)
	:
	robot_control(hardware.ip),
	controller_(ip)
{
}
;

void robot_control_hardware::move_to_configuration(const franka_proxy::robot_config_7dof& destination)
{
	bool move_completed = false;
	while (!move_completed)
	{
		try
		{
			move_without_errors_to_configuration(destination);
			move_completed = true;
		}
		catch (const std::exception& e)
		{
			std::cout << e.what() << std::endl;
			controller_.automatic_error_recovery();
		}
	}

}


franka_proxy::robot_config_7dof robot_control_hardware::get_current_config()
{
	return controller_.current_config();
}



void robot_control_hardware::update()
{
	controller_.update();
	
}

void robot_control_hardware::move_without_errors_to_configuration(const franka_proxy::robot_config_7dof& destination)
{
	controller_.move_to(destination);
}

Eigen::Vector3d robot_control_hardware::get_nsa_position(const franka_proxy::robot_config_7dof& current_pos)
{
	std::vector<Eigen::Affine3d> positions = franka_proxy::franka_proxy_util::fk(current_pos);
	int count = 0;

	return positions[7].translation();
}

Eigen::Affine3d robot_control_hardware::get_nsa_frame(const franka_proxy::robot_config_7dof& current_pos)
{
	return franka_proxy::franka_proxy_util::fk(current_pos)[7];
}

void robot_control_hardware::get_robot_coordinates_for_hand()
{
	// get positions
	const franka_proxy::robot_config_7dof pose = get_current_config();

	// vectors for transformation calculation
	std::vector<Eigen::Vector3d> world_points;

	// define coordinates of a hand
	std::ifstream in("coordinates.txt");
	
	std::vector<Eigen::Vector3d> landmarks;
	
	for (int i = 0; i < 21; ++i) {
		float x, y, z;

		in >> x >> y >> z;
		landmarks.push_back(Eigen::Vector3d(x, y, z));
	}

	auto nsa = robot_control_hardware::get_nsa_frame(pose);
	auto point = robot_control_hardware::get_nsa_position(pose);

	std::cout << nsa.linear() << std::endl;

	for (const Eigen::Vector3d& landmark : landmarks)
	{
		auto coordinates = point + nsa.linear() * landmark;
		world_points.push_back(coordinates);
	}

	// Output to CSV file
	std::ofstream out("landmarks.csv");
	out << "Robot Joints:";
	for (const auto& angle : pose)
		out << " " << angle;
	out << std::endl;

	out << "x y z";
	for (const Eigen::Vector3d& landmark : world_points)
		out << std::endl << landmark.x() << " " << landmark.y() << " " << landmark.z();
}
