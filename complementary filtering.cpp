void IMUupdate(double gx, double gy, double gz, double ax, double ay, double az, double alpha, float dt)
{
    double norm;
    double vx, vy, vz;
    double ex, ey, ez; 
    double exInt,eyInt,ezInt;
    double fusion_angular_rate;

    // 将角速度转换为弧度/秒  
    gx = gx * 0.01745329f;
    gy = gy * 0.01745329f;
    gz = gz * 0.01745329f;

    // 归一化加速度计测量值
    norm = sqrt(ax * ax + ay * ay + az * az);
    ax = ax / norm;
    ay = ay / norm;
    az = az / norm;

    // 估计重力方向
    vx = 2.0f * (q1 * q3 - q0 * q2);
    vy = 2.0f * (q0 * q1 + q2 * q3);
    vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;

    // 计算误差项
    ex = (ay * vz - az * vy);
    ey = (az * vx - ax * vz);
    ez = (ax * vy - ay * vx);

    // 误差项积分
    exInt = exInt + ex * Ki * dt;
    eyInt = eyInt + ey * Ki * dt;
    ezInt = ezInt + ez * Ki * dt;

    // 调整陀螺仪测量值，对每个轴进行互补滤波
    gx = alpha * (gx) + (1 - alpha) * (Kp * (*ex) + (*exInt));
    gy = alpha * (gy) + (1 - alpha) * (Kp * (*ey) + (*eyInt));
    gz = alpha * (gz) + (1 - alpha) * (Kp * (*ez) + (*ezInt));

    // 四元数积分更新
    q0 = q0 + (-q1 * gx - q2 * gy - q3 * gz) * 0.5f * dt;
    q1 = q1 + ( q0 * gx + q2 * gz - q3 * gy) * 0.5f * dt;
    q2 = q2 + ( q0 * gy - q1 * gz + q3 * gx) * 0.5f * dt;
    q3 = q3 + ( q0 * gz + q1 * gy - q2 * gx) * 0.5f * dt;

    // 四元数归一化
    norm = sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    q0 = q0 / norm;
    q1 = q1 / norm;
    q2 = q2 / norm;
    q3 = q3 / norm;

    // 欧拉角计算
    pitch = asin(-2 * q1 * q3 + 2 * q0 * q2) * 57.29577951f; 
    roll = atan2(2 * q2 * q3 + 2 * q0 * q1, -2 * q1 * q1 - 2 * q2 * q2 + 1) * 57.29577951f; 
    yaw = atan2(2 * (q1 * q2 + q0 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * 57.29577951f;
}
