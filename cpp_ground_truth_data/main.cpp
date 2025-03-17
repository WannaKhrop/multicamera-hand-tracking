#pragma once

#include <iostream>
#include <Windows.h>
#include <Eigen/Geometry>

#include "hardware/robot_control_hardware.hpp"

int main()
{
	robot_control_hardware control("132.180.194.120");

	control.get_robot_coordinates_for_hand();
	
	return 0;
}
