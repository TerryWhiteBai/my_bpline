cmake_minimum_required(VERSION 3.16)
project(marvin_description VERSION 0.1.0 LANGUAGES CXX)

find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_marvin_description test/test_marvin_description.py)
endif()

install(DIRECTORY config urdf DESTINATION share/${PROJECT_NAME})
install(
  DIRECTORY docs
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)
install(
  DIRECTORY launch rviz
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)
install(DIRECTORY meshes/ DESTINATION share/${PROJECT_NAME}/meshes)

ament_package()
