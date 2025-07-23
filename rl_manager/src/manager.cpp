#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "gazebo_msgs/msg/model_state.hpp"


using namespace std::chrono_literals;
using std::placeholders::_1;


class RLManager : public rclcpp::Node
{
public:
    RLManager(): Node("rl_manager")
    {

        model_state_subscriber_ = this->create_subscription<gazebo_msgs::msg::ModelState>(
            "model_states",
            10, std::bind(&RLManager::model_state_callback, this, _1));
            


        image_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/static_camera/static_camera/image_raw",
            10, std::bind(&RLManager::image_callback, this, _1));

        publisher_ = this->create_publisher<std_msgs::msg::String>("rl_topic", 10);
        timer_ = this->create_wall_timer(500ms, std::bind(&RLManager::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello from RLManager";
        publisher_->publish(message);
    }



    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // RCLCPP_ERROR(this->get_logger(), "Image callback not implemented yet");
        RCLCPP_INFO(this->get_logger(), "Received image with width: %d, height: %d", msg->width, msg->height);
        // Process the image as needed
    }
    void model_state_callback(const gazebo_msgs::msg::ModelState::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received model state for model: %s", msg->model_name.c_str());
        // Process the model state as needed
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Subscription<gazebo_msgs::msg::ModelState>::SharedPtr model_state_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;

};


int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RLManager>());
    rclcpp::shutdown();
    return 0;
}