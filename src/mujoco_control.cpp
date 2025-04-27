// mujoco_controller.cpp
#include <zmq.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <json/json.h>

class MujocoController {
public:
    MujocoController(const std::string& state_addr = "tcp://localhost:5556", 
                    const std::string& cmd_addr = "tcp://localhost:5555")
        : ctx_(1), 
          state_sub_(ctx_, ZMQ_SUB),
          cmd_pub_(ctx_, ZMQ_PUB) {
        
        // 连接状态订阅
        state_sub_.connect(state_addr);
        state_sub_.setsockopt(ZMQ_SUBSCRIBE, "", 0);
        
        // 连接控制发布
        cmd_pub_.connect(cmd_addr);
    }

    Json::Value getState(int timeout_ms = 1000) {
        zmq::pollitem_t items[] = {{state_sub_, 0, ZMQ_POLLIN, 0}};
        zmq::poll(items, 1, timeout_ms);
        
        if (items[0].revents & ZMQ_POLLIN) {
            zmq::message_t msg;
            state_sub_.recv(&msg);
            
            Json::Value state;
            Json::Reader reader;
            if (reader.parse(static_cast<char*>(msg.data()), state)) {
                return state;
            }
        }
        return Json::nullValue;
    }

    void sendControl(const std::vector<double>& cmd) {
        Json::Value json_cmd;
        json_cmd["control_cmd"] = Json::Value(Json::arrayValue);
        for (double val : cmd) {
            json_cmd["control_cmd"].append(val);
        }

        Json::FastWriter writer;
        std::string msg_str = writer.write(json_cmd);
        
        zmq::message_t msg(msg_str.begin(), msg_str.end());
        cmd_pub_.send(msg);
    }

private:
    zmq::context_t ctx_;
    zmq::socket_t state_sub_;
    zmq::socket_t cmd_pub_;
};

int main() {
    try {
        MujocoController controller;
        
        // 获取状态
        auto state = controller.getState();
        std::cout << "Current state: " << state << std::endl;
        
        // 发送控制指令 (示例: 12个关节扭矩为0)
        std::vector<double> cmd(12, 0.0);
        controller.sendControl(cmd);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}