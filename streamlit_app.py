import streamlit as st
import os
from app import rag_query

# 页面配置
st.set_page_config(
    page_title="Ariba RAG 问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {'sourcing': [], 'integration': []}
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'sourcing'

# 侧边栏配置
with st.sidebar:
    st.title("📚 Ariba RAG 系统")
    st.markdown("---")

    # Tab选择（使用radio buttons）
    query_type = st.radio(
        "选择文档类型",
        options=["Sourcing", "Integration"],
        index=0,
        key="query_type_selector"
    )

    # 转换为小写用于后端调用
    current_type = query_type.lower()
    st.session_state.current_tab = current_type

    st.markdown("---")

    # 高级设置（可折叠）
    with st.expander("⚙️ 高级设置", expanded=False):
        n_results = st.slider("检索文档数量", 3, 10, 5, key="n_results_slider")
        temperature = st.slider("LLM温度", 0.0, 1.0, 0.3, 0.1, key="temperature_slider")

    # 显示文档详情选项（放在expander外面，确保全局可访问）
    show_metadata = st.checkbox("显示文档详情", value=False, key="show_metadata_checkbox")

    st.markdown("---")

    # 清空历史按钮
    if st.button("🗑️ 清空当前对话历史"):
        st.session_state.chat_history[current_type] = []
        st.rerun()

    # 系统信息
    st.markdown("---")
    st.caption("💡 提示：")
    st.caption("- 输入问题后按Enter或点击发送")
    st.caption("- 支持中英文问答")
    st.caption("- 答案基于Ariba官方文档")

# 主标题
st.title(f"💬 {query_type} 问答")
st.markdown(f"当前模式：**{query_type}** 文档")

# 显示对话历史
chat_container = st.container()
with chat_container:
    for qa in st.session_state.chat_history[current_type]:
        # 用户问题
        with st.chat_message("user"):
            st.markdown(qa['question'])

        # 系统回答
        with st.chat_message("assistant"):
            st.markdown(qa['answer'])

            # 显示引用来源
            if qa.get('sources'):
                with st.expander("📄 参考文档来源"):
                    for i, source in enumerate(qa['sources'], 1):
                        st.markdown(f"{i}. {source}")

            # 显示文档详情（可选）
            if show_metadata and qa.get('retrieved_docs'):
                with st.expander("🔍 检索文档详情"):
                    for i, doc in enumerate(qa['retrieved_docs'], 1):
                        if isinstance(doc, dict):
                            meta = doc.get('metadata', {})
                            st.markdown(f"**文档 {i}:**")
                            st.markdown(f"- 标题: {meta.get('document_title', 'N/A')}")
                            st.markdown(f"- 页码: 第{meta.get('page', 0) + 1}页")
                            if 'rerank_score' in doc:
                                st.markdown(f"- 相关性: {doc['rerank_score']:.4f}")
                            st.markdown(f"- 内容预览: {doc.get('content', '')[:200]}...")
                            st.markdown("---")

# 输入框（固定在底部）
user_input = st.chat_input("请输入您的问题...")

if user_input:
    # 检查API key
    if not os.getenv("DASHSCOPE_API_KEY"):
        st.error("❌ 请设置DASHSCOPE_API_KEY环境变量")
        st.stop()

    # 显示用户输入
    with st.chat_message("user"):
        st.markdown(user_input)

    # 显示加载状态
    with st.chat_message("assistant"):
        with st.spinner("🤔 正在思考..."):
            try:
                # 调用RAG系统
                result = rag_query(
                    query_text=user_input,
                    type_of_query=current_type,
                    n_results=n_results,
                    temperature=temperature
                )

                # 显示答案
                st.markdown(result['answer'])

                # 显示引用来源
                if result.get('sources'):
                    with st.expander("📄 参考文档来源"):
                        for i, source in enumerate(result['sources'], 1):
                            st.markdown(f"{i}. {source}")

                # 显示文档详情（可选）
                if show_metadata and result.get('retrieved_docs'):
                    with st.expander("🔍 检索文档详情"):
                        for i, doc in enumerate(result['retrieved_docs'], 1):
                            if isinstance(doc, dict):
                                meta = doc.get('metadata', {})
                                st.markdown(f"**文档 {i}:**")
                                st.markdown(f"- 标题: {meta.get('document_title', 'N/A')}")
                                st.markdown(f"- 页码: 第{meta.get('page', 0) + 1}页")
                                if 'rerank_score' in doc:
                                    st.markdown(f"- 相关性: {doc['rerank_score']:.4f}")
                                st.markdown(f"- 内容预览: {doc.get('content', '')[:200]}...")
                                st.markdown("---")

                # 保存到历史记录
                st.session_state.chat_history[current_type].append({
                    'question': user_input,
                    'answer': result['answer'],
                    'sources': result.get('sources', []),
                    'retrieved_docs': result.get('retrieved_docs', [])
                })

            except Exception as e:
                st.error(f"❌ 查询失败: {str(e)}")
                st.exception(e)
