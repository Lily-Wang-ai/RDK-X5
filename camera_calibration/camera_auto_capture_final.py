import sys, os, time
import cv2
import numpy as np
from hobot_vio import libsrcampy
import threading
import queue


def calculate_frame_difference(frame1, frame2):
    """计算两帧之间的差异"""
    if frame1 is None or frame2 is None:
        return float('inf')
    
    # 转换为灰度图
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # 计算帧差
    diff = cv2.absdiff(gray1, gray2)
    
    # 计算差异的平均值
    mean_diff = np.mean(diff)
    
    return mean_diff

def detect_motion_stability(frame_buffer, stability_threshold=5.0, min_frames=10):
    """检测画面是否稳定（无明显运动）"""
    if len(frame_buffer) < min_frames:
        return False, 0.0
    
    # 检查最近几帧的稳定性
    recent_frames = frame_buffer[-min_frames:]
    differences = []
    
    for i in range(1, len(recent_frames)):
        diff = calculate_frame_difference(recent_frames[i-1], recent_frames[i])
        differences.append(diff)
    
    # 如果所有差异都小于阈值，认为画面稳定
    avg_diff = np.mean(differences)
    max_diff = np.max(differences)
    
    # 只在需要时打印调试信息
    # print(f"画面稳定性检测: 平均差异={avg_diff:.2f}, 最大差异={max_diff:.2f}, 阈值={stability_threshold}")
    
    # 更宽松的稳定性判断：只要平均差异小于阈值就认为稳定
    is_stable = avg_diff < stability_threshold  # 移除了max_diff的严格要求
    return is_stable, avg_diff

def test_camera(output_dir="captured_images"):
    """
    相机自动拍摄标定板功能 - 检测画面稳定时自动拍照
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
   
    # screen_width, screen_height = 1920, 1080  # 屏幕分辨率
    


    cam = libsrcampy.Camera()
    
    # 暂时禁用Display功能以避免分段错误
    display_available = False
    print("✗ Display功能已禁用（避免分段错误），使用文件预览模式")
    
    # 相机参数
    width, height = 1920, 1080

    #open MIPI camera, fps: 30, solution: 1080p
    ret = cam.open_cam(0, -1, -1, 1920, 1080, height, width)  # 注意sensor_height和sensor_width的顺序
    print("Camera open_cam return:%d" % ret)

    if ret != 0:
        print("相机打开失败!")
        return

    # wait for 1s
    time.sleep(1)
    
    print("=== 标定板自动拍摄系统 ===")
    print("功能说明:")
    print("- 自动检测: 当画面稳定时自动拍照")
    print("- 手动拍照: 按回车键立即拍照")
    print("- 实时显示: 拍摄后短暂显示图片")
    print("- 退出程序: 输入 'q' + 回车")
    print("- 强制退出: Ctrl+C")
    print()
    
    # 拍摄参数 - 调整这些值来减小稳定性要求
    auto_frame_count = 0  # 自动拍摄计数器
    manual_frame_count = 0  # 手动拍摄计数器
    frame_buffer = []  # 存储最近的帧用于稳定性检测
    max_buffer_size = 8   # 减少到8帧，更快触发检测（原10帧）
    stability_threshold = 10.0  # 放宽稳定性阈值，更容易触发（原5.0，值越大越宽松）
    stable_duration = 1.0  # 减少稳定持续时间到1秒（原1.5秒）
    min_capture_interval = 1.5  # 减少最小拍摄间隔到1.5秒（原2秒）
    
    # 显示相关参数
    enable_preview = True  # 是否启用拍摄后预览
    preview_duration = 2000  # 预览显示时间（毫秒）
    
    last_capture_time = 0
    stable_start_time = None
    current_frame = None
    auto_mode = True  # 是否启用自动拍摄
    
    # 用于线程间通信的队列
    command_queue = queue.Queue()
    
    def input_thread():
        """处理用户输入的线程"""
        nonlocal auto_mode
        while True:
            try:
                user_input = input().strip().lower()
                command_queue.put(user_input)
                if user_input == 'q':
                    break
            except:
                break
    
    # 启动输入线程
    input_handler = threading.Thread(target=input_thread, daemon=True)
    input_handler.start()
    
    print(f"参数设置:")
    print(f"- 稳定性阈值: {stability_threshold}")
    print(f"- 稳定持续时间: {stable_duration}秒")
    print(f"- 最小拍摄间隔: {min_capture_interval}秒")
    print(f"- 自动拍摄: {'开启' if auto_mode else '关闭'}")
    print(f"- 拍摄预览: {'开启' if enable_preview else '关闭'}")
    print()
    print("开始实时监控，等待稳定画面...")
    print("可输入命令: 回车=立即拍照, 'a'=切换自动模式, 'p'=切换预览, 'q'=退出")
    print("-" * 60)
    
    try:
        while True:
            # 持续获取实时图像
            nv12_img = cam.get_img(2, width, height)
            current_time = time.time()
            
            if nv12_img is not None:
                # 转换NV12为BGR格式
                yuv_img = np.frombuffer(nv12_img, dtype=np.uint8).reshape(height * 3 // 2, width)
                bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR_NV12)
                
                # 更新当前实时画面
                current_frame = bgr_img.copy()

                # # 居中显示图像
                # w,h=current_frame.shape[:2]
                # print(f"获取图像成功: {w}x{h}, 时间戳: {current_time:.2f}")#1080*1920
                # x = (screen_width - w) // 2
                # y = (screen_height - h) // 2

                # 添加到帧缓冲区
                frame_buffer.append(current_frame)
                if len(frame_buffer) > max_buffer_size:
                    frame_buffer.pop(0)
                
                # 检测画面稳定性
                is_stable = False
                avg_diff = 0.0
                if auto_mode:
                    # print(f"调试: 帧缓冲区大小={len(frame_buffer)}, 需要={max_buffer_size}")
                    if len(frame_buffer) >= max_buffer_size:
                        is_stable, avg_diff = detect_motion_stability(frame_buffer, stability_threshold, max_buffer_size//2)
                        # print(f"调试: 稳定检测结果={is_stable}, 差异={avg_diff:.2f}")
                        
                        if is_stable:
                            if stable_start_time is None:
                                stable_start_time = current_time
                                print(f"检测到画面开始稳定... (差异={avg_diff:.2f})")
                            else:
                                stable_time = current_time - stable_start_time
                                # print(f"调试: 稳定时间={stable_time:.1f}, 需要={stable_duration}")
                                if stable_time >= stable_duration:
                                    # 画面已稳定足够时间，检查是否可以拍照
                                    # print(f"调试: 距离上次拍摄={current_time - last_capture_time:.1f}, 需要={min_capture_interval}")
                                    if current_time - last_capture_time >= min_capture_interval:
                                        print(f"\n*** 画面稳定 {stable_time:.1f}秒，自动拍摄! ***")
                                        
                                        # 自动拍摄
                                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                                        filename = f"calibration_auto_{timestamp}_{auto_frame_count:04d}.jpg"
                                        filepath = os.path.join(output_dir, filename)
                                        
                                        success = cv2.imwrite(filepath, current_frame)
                                        if success:
                                            # 显示图片
                                             #居中显示窗口设置
                                            window_name = 'cam_Image Preview'
                                            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                                            cv2.moveWindow(window_name, 0,0)#移动窗口到屏幕中心
                                            cv2.imshow(window_name, current_frame)
                                            cv2.waitKey(1000)  # 显示图片，等待1毫秒
                                            cv2.destroyAllWindows()  # 关闭窗口
                                            
                                            file_size = os.path.getsize(filepath)
                                            print(f"✓ 自动拍摄成功: {filename} ({file_size} bytes)")
                                            auto_frame_count += 1
                                            last_capture_time = current_time
                                            stable_start_time = None  # 重置稳定计时
                                            
                                            # # 显示拍摄预览
                                            # if enable_preview:
                                            #     preview_text = f"AUTO SHOT #{auto_frame_count}"
                                            #     # 保存预览图片文件
                                            #     save_preview_image(current_frame, filepath, preview_text)
                                            #     # 尝试OpenCV显示
                                            #     preview_img = create_preview_image(current_frame, preview_text)
                                            #     if not safe_imshow("Auto Capture Preview", preview_img, preview_duration):
                                            #         print("注意: OpenCV预览显示功能不可用")
                                        else:
                                            print("✗ 自动拍摄失败!")
                                    else:
                                        remaining = min_capture_interval - (current_time - last_capture_time)
                                        print(f"画面稳定但需等待 {remaining:.1f}秒后才能拍摄")
                                else:
                                    # 显示稳定进度（每0.5秒显示一次）
                                    if int(stable_time * 2) != int((stable_time - 0.033) * 2):
                                        print(f"画面保持稳定中... {stable_time:.1f}/{stable_duration}秒")
                        else:
                            if stable_start_time is not None:
                                print(f"画面不再稳定 (差异={avg_diff:.2f})，重置检测...")
                                stable_start_time = None
                    # else:
                    #     print(f"调试: 等待更多帧数据... {len(frame_buffer)}/{max_buffer_size}")
                
                # 处理用户输入命令
                try:
                    user_input = command_queue.get_nowait()
                    
                    if user_input == 'q':
                        print("退出程序...")
                        break
                    elif user_input == 'a':
                        auto_mode = not auto_mode
                        print(f"自动拍摄模式: {'开启' if auto_mode else '关闭'}")
                        stable_start_time = None  # 重置稳定计时
                    elif user_input == 'p':
                        enable_preview = not enable_preview
                        print(f"拍摄预览功能: {'开启' if enable_preview else '关闭'}")
                    elif user_input == '' or user_input == 'c':  # 回车键或c键
                        # 手动拍摄
                        if current_frame is not None:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"calibration_manual_{timestamp}_{manual_frame_count:04d}.jpg"
                            filepath = os.path.join(output_dir, filename)
                            
                            success = cv2.imwrite(filepath, current_frame)
                            if success:
                                file_size = os.path.getsize(filepath)
                                print(f"✓ 手动拍摄成功: {filename} ({file_size} bytes)")
                                manual_frame_count += 1
                                last_capture_time = current_time
                                
                                # 显示拍摄预览
                                if enable_preview:
                                    preview_text = f"MANUAL SHOT #{manual_frame_count}"
                                    # 保存预览图片文件
                                    save_preview_image(current_frame, filepath, preview_text)
                                    # 尝试OpenCV显示
                                    preview_img = create_preview_image(current_frame, preview_text)
                                    if not safe_imshow("Manual Capture Preview", preview_img, preview_duration):
                                        print("注意: OpenCV预览显示功能不可用")
                                
                                # # 保存原始NV12数据
                                # nv12_filename = f"calibration_manual_{timestamp}_{manual_frame_count-1:04d}.img"
                                # nv12_filepath = os.path.join(output_dir, nv12_filename)
                                # with open(nv12_filepath, "wb") as fo:
                                #     fo.write(nv12_img)
                                # print(f"✓ 原始数据已保存: {nv12_filename}")
                            else:
                                print("✗ 手动拍摄失败!")
                        else:
                            print("✗ 当前没有可用的画面!")
                    elif user_input.startswith('t'):
                        # 调整稳定性阈值
                        try:
                            new_threshold = float(user_input[1:]) if len(user_input) > 1 else stability_threshold
                            stability_threshold = new_threshold
                            print(f"稳定性阈值已调整为: {stability_threshold}")
                        except:
                            print(f"当前稳定性阈值: {stability_threshold}")
                    else:
                        print("可用命令: 回车=拍照, 'a'=切换自动, 'p'=切换预览, 't数值'=调整阈值, 'q'=退出")
                        
                except queue.Empty:
                    # 没有用户输入，继续处理
                    pass
                
                # 定期显示状态信息
                if int(current_time) % 5 == 0 and int(current_time * 10) % 10 == 0:
                    status = "自动监控中" if auto_mode else "手动模式"
                    if stable_start_time:
                        stable_info = f"稳定{current_time - stable_start_time:.1f}s"
                    else:
                        stable_info = f"检测中(差异={avg_diff:.2f})" if avg_diff > 0 else "等待帧数据"
                    total_shots = auto_frame_count + manual_frame_count
                    print(f"[状态] {status} | 总拍摄: {total_shots}张 (自动:{auto_frame_count}, 手动:{manual_frame_count}) | 画面: {stable_info}")
            
            else:
                print("获取图像失败!")
                time.sleep(0.1)
            
            # 控制帧率，避免CPU过度使用
            time.sleep(0.033)  # 约30fps

    except KeyboardInterrupt:
        print("\n程序被中断...")
    
    finally:
        # 释放资源
        try:
            cv2.destroyAllWindows()  # 确保关闭所有OpenCV窗口
        except:
            pass
        
        cam.close_cam()
        print("\n" + "=" * 60)
        print(f"标定板拍摄完成!")
        total_shots = auto_frame_count + manual_frame_count
        print(f"总共拍摄了 {total_shots} 张图像，保存到: {output_dir}")
        print(f"  - 自动拍摄: {auto_frame_count} 张")
        print(f"  - 手动拍摄: {manual_frame_count} 张")
        print(f"文件命名规则:")
        print(f"- 自动拍摄: calibration_auto_时间戳_序号.jpg")
        print(f"- 手动拍摄: calibration_manual_时间戳_序号.jpg")
        if enable_preview:
            print(f"- 预览图片: 原文件名_preview.jpg")
        print("camera_auto_capture done!!!")

if __name__ == "__main__":
    output_dir = "calibration_images"  # 专门用于标定的目录
    test_camera(output_dir)