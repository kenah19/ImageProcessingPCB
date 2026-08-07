# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pcb_engine import detect_pcb_defects

st.set_page_config(page_title="传统算法-PCB缺陷检测系统", layout="wide")

st.title("📟 传统图像处理 - PCB 缺陷检测系统")
st.write("本系统拒绝深度学习，采用 **ORB 特征配准 + Otsu 二值化 + 图像物理差分法** 提取 PCB 板工艺缺陷。")

st.info("💡 **提示：** 本系统需要上传两张图片：一幅为【无缺陷模板图】，另一幅为【待测缺陷图】。")

# 侧边栏
st.sidebar.header("🔧 传统算法微调参数")
min_area = st.sidebar.slider("过滤微小噪点面积 (Min Area)", 2, 50, 15, step=1)

col_upload1, col_upload2 = st.columns(2)
with col_upload1:
    temp_file = st.file_uploader("1. 上传无缺陷模板图 (Template)", type=["jpg", "png", "jpeg"])
with col_upload2:
    defect_file = st.file_uploader("2. 上传待测缺陷图 (Defect Image)", type=["jpg", "png", "jpeg"])

if temp_file is not None and defect_file is not None:
    # 转换 PIL 为 OpenCV 格式 (numpy RGB)
    temp_img = np.array(Image.open(temp_file))
    def_img = np.array(Image.open(defect_file))
    
    # 确保是 3 通道的 RGB
    if len(temp_img.shape) == 2:
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2RGB)
    if len(def_img.shape) == 2:
        def_img = cv2.cvtColor(def_img, cv2.COLOR_GRAY2RGB)

    # 运行算法
    with st.spinner("正在进行图像特征配准与像素级差分计算..."):
        # 调用我们自己写的传统 CV 引擎
        output_img, diff_mask, count = detect_pcb_defects(temp_img, def_img)

    # 布局一：展示输入和输出
    st.markdown("---")
    st.subheader("🎯 实时检测结果可视化")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(temp_img, caption="无缺陷模板图 (Standard)", use_container_width=True)
    with c2:
        st.image(def_img, caption="待检测原图 (Raw Defect)", use_container_width=True)
    with c3:
        st.image(output_img, caption="算法检出结果 (Detected)", use_container_width=True)

    # 布局二：传统算法过程拆解（答辩加分神器！）
    st.markdown("---")
    st.subheader("🔬 计算机视觉中间级运算过程（算法透明化展示）")
    st.write("传统视觉相比深度学习的最大优势在于**可解释性强**。下面是算法运行的中间数据流：")
    
    c4, c5 = st.columns(2)
    with c4:
        # 展示配准差异二值图
        st.image(diff_mask, caption="像素异或差异提取图 (XOR Mask)", use_container_width=True, clamp=True)
        st.write("ℹ️ 黑色代表两图完全一致，白色代表像素级别存在突变（即潜在缺陷）。")
    with c5:
        st.metric(label="📊 检出缺陷连通域总数", value=f"{count} 处")
        if count > 0:
            st.error("🚨 检测结论：该 PCB 板未通过质检，存在断路/短路/毛刺等局部物理形变！")
        else:
            st.success("🎉 检测结论：经过完美像素比对，该 PCB 板与标准件完全一致，质量合格！")