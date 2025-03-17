#include "robot_control.hpp"

#include <iostream>


robot_control::robot_control(const std::string& proxy_ip) :
	robot_updater(),
	ip(proxy_ip)
{
};

franka_control::robot_config_7dof robot_control::change_pose_to_franka_control(const franka_proxy::robot_config_7dof& in)
{
	return franka_control::robot_config_7dof{ {in[0], in[1], in[2], in[3], in[4], in[5], in[6]} };
}

franka_proxy::robot_config_7dof robot_control::change_pose_to_franka_proxy(const franka_control::robot_config_7dof& in)
{
	return franka_proxy::robot_config_7dof{ {in[0], in[1], in[2], in[3], in[4], in[5], in[6]} };
}

franka_control::robot_config_7dof robot_control::calculate_joints_from_cartesian(const Eigen::Affine3d& nsa)
{
	auto solution(franka_proxy::franka_proxy_util::ik_fast(nsa));

	if (solution.empty())
	{
		throw std::exception("no solution found");
	}
	else
	{
		return solution.front();
	}
}

robot_updater::robot_updater()
	:
	is_running_(false),
	thread_()
{
}

void robot_updater::start_thread()
{
	if (is_running_)
		return;
	is_running_ = true;
	thread_ = std::thread(&robot_updater::update_loop, this);

	std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void robot_updater::stop_thread()
{
	if (!is_running_)
		return;
	is_running_ = false;
	thread_.join();
}

robot_updater::~robot_updater()
{
	if (!is_running_)
		return;
	is_running_ = false;
	thread_.join();
}

void robot_updater::update_loop()
{
	std::cout << "Call: " << is_running_ << std::endl;
	while (is_running_)
	{
		try
		{
			update();
		}
		catch (const std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(waiting_time));
	}
}
