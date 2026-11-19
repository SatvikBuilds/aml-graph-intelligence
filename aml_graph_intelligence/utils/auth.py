"""
Authentication and Role-Based Access Control Module
Provides login, registration, session persistence, and MongoDB status management.
"""

import streamlit as st
from datetime import datetime
from database.mongo_manager import get_mongo_manager

ROLES = [
    "Chief AML Officer",
    "Senior AML Investigator",
    "Fraud Forensic Analyst",
    "Regulatory Compliance Auditor",
    "FIU Reporting Specialist"
]

def init_auth_state():
    """Initializes authentication variables in session state."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "current_user" not in st.session_state:
        st.session_state["current_user"] = None

def is_authenticated() -> bool:
    """Returns True if officer is currently signed in."""
    init_auth_state()
    return st.session_state.get("authenticated", False)

def get_current_user() -> dict:
    """Returns the authenticated user dict or empty dict."""
    init_auth_state()
    return st.session_state.get("current_user") or {}

def logout_user():
    """Logs out officer and clears session state."""
    mongo = get_mongo_manager()
    user = get_current_user()
    if user:
        mongo.log_audit(user.get("username", "USER"), user.get("role", "OFFICER"), "LOGOUT", "AUTH", "Officer signed out.")
    st.session_state["authenticated"] = False
    st.session_state["current_user"] = None
    st.rerun()

def render_login_screen():
    """Renders the dark navy GraphAML Login & Database Configuration Portal."""
    mongo = get_mongo_manager()
    status = mongo.get_status()

    # Center layout
    _, center_col, _ = st.columns([1, 2.2, 1])

    with center_col:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 24px;">
            <div style="display: inline-flex; align-items: center; justify-content: center; width: 68px; height: 68px; background: linear-gradient(135deg, #0284c7, #06b6d4); border-radius: 16px; font-size: 32px; box-shadow: 0 0 25px rgba(6, 182, 212, 0.4); margin-bottom: 12px;">
                🛡️
            </div>
            <h1 style="margin: 0; font-size: 2rem; color: #f8fafc; font-weight: 700;">GraphAML Surveillance Portal</h1>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">
                Dynamic Graph Intelligence for Multi-Hop Anti-Money Laundering Detection
            </p>
        </div>
        """, unsafe_allow_html=True)

        # MongoDB Status Pill
        if status["is_connected"]:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid #10b981; border-radius: 8px; padding: 8px 14px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem;">
                <span style="color: #34d399; font-weight: 600;">● MongoDB Active: <code>{status['db_name']}</code> ({status['ping_ms']}ms)</span>
                <span style="color: #94a3b8; font-family: monospace;">{status['masked_uri']}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid #f59e0b; border-radius: 8px; padding: 8px 14px; margin-bottom: 18px; font-size: 0.85rem;">
                <span style="color: #fbbf24; font-weight: 600;">⚠️ MongoDB Standby (Local Memory Fallback Active)</span>
                <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 3px;">Configure your connection string below if connecting to a remote Atlas instance.</div>
            </div>
            """, unsafe_allow_html=True)

        tab_login, tab_register, tab_mongo = st.tabs([
            "🔐 Officer Sign In", 
            "📝 Register Investigator", 
            "🍃 MongoDB Settings"
        ])

        # TAB 1: SIGN IN
        with tab_login:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            with st.form("login_form"):
                uname = st.text_input("Investigator Username", placeholder="e.g. admin or investigator")
                pwd = st.text_input("Security Passcode", type="password", placeholder="Enter passcode")
                submit_login = st.form_submit_button("Authenticate & Enter System", use_container_width=True)

            if submit_login:
                if not uname or not pwd:
                    st.error("Please enter both username and password.")
                else:
                    success, user_data, msg = mongo.authenticate_user(uname, pwd)
                    if success:
                        st.session_state["authenticated"] = True
                        st.session_state["current_user"] = user_data
                        st.success(f"Welcome back, {user_data['full_name']}!")
                        st.rerun()
                    else:
                        st.error(msg)

            # Quick Demo Credentials
            st.markdown("---")
            st.markdown("<span style='font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;'>⚡ One-Click Demo Credentials:</span>", unsafe_allow_html=True)
            
            c_d1, c_d2, c_d3 = st.columns(3)
            with c_d1:
                if st.button("👑 Chief Officer", use_container_width=True):
                    success, user_data, _ = mongo.authenticate_user("admin", "admin123")
                    if success:
                        st.session_state["authenticated"] = True
                        st.session_state["current_user"] = user_data
                        st.rerun()
            with c_d2:
                if st.button("🕵️ Senior Analyst", use_container_width=True):
                    success, user_data, _ = mongo.authenticate_user("investigator", "aml2025")
                    if success:
                        st.session_state["authenticated"] = True
                        st.session_state["current_user"] = user_data
                        st.rerun()
            with c_d3:
                if st.button("📋 Auditor", use_container_width=True):
                    success, user_data, _ = mongo.authenticate_user("auditor", "audit2025")
                    if success:
                        st.session_state["authenticated"] = True
                        st.session_state["current_user"] = user_data
                        st.rerun()

        # TAB 2: REGISTER
        with tab_register:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            with st.form("register_form"):
                r_full_name = st.text_input("Full Official Name", placeholder="e.g. Special Agent John Doe")
                r_uname = st.text_input("Chosen Username", placeholder="e.g. jdoe_aml")
                r_role = st.selectbox("Designated Compliance Role", ROLES)
                r_pwd = st.text_input("New Password (min 6 characters)", type="password")
                r_pwd_confirm = st.text_input("Confirm Password", type="password")
                submit_reg = st.form_submit_button("Register Officer Entity", use_container_width=True)

            if submit_reg:
                if not r_full_name or not r_uname or not r_pwd:
                    st.error("Please fill in all registration fields.")
                elif r_pwd != r_pwd_confirm:
                    st.error("Passwords do not match.")
                else:
                    reg_ok, reg_msg = mongo.register_user(r_uname, r_pwd, r_full_name, r_role)
                    if reg_ok:
                        st.success(reg_msg)
                        st.info("You can now switch to the 'Officer Sign In' tab to authenticate.")
                    else:
                        st.error(reg_msg)

        # TAB 3: MONGODB CONFIGURATION
        with tab_mongo:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 12px;">
                Connect to a local MongoDB instance (<code>mongodb://localhost:27017/</code>) or a 
                cloud MongoDB Atlas cluster (<code>mongodb+srv://...</code>).
            </div>
            """, unsafe_allow_html=True)

            custom_uri = st.text_input(
                "MongoDB Connection URI", 
                value=status.get("uri", "mongodb://localhost:27017/"),
                type="default"
            )
            custom_db = st.text_input("Database Name", value=status.get("db_name", "aml_graph_intelligence"))

            if st.button("🔌 Test & Update MongoDB Connection", use_container_width=True):
                with st.spinner("Pinging MongoDB cluster..."):
                    connected = mongo.connect(custom_uri, custom_db)
                    if connected:
                        st.success(f"✅ Connected to MongoDB at '{custom_uri}'! Database: '{custom_db}'")
                        st.rerun()
                    else:
                        st.error(f"Failed to connect: {mongo.connection_error}")

            # Diagnostics display
            if status["is_connected"]:
                st.markdown("#### Database Collections Overview")
                st.json(status.get("doc_counts", {}))

def require_authentication():
    """
    Guards pages against unauthenticated access.
    Renders login screen and stops further rendering if not signed in.
    """
    init_auth_state()
    if not st.session_state["authenticated"]:
        render_login_screen()
        st.stop()

def render_officer_sidebar():
    """Renders the officer credentials badge and MongoDB health indicator in sidebar."""
    user = get_current_user()
    mongo = get_mongo_manager()
    status = mongo.get_status()

    if user:
        st.sidebar.markdown(f"""
        <div style="background: linear-gradient(145deg, #0f172a, #1e293b); border: 1px solid #334155; border-radius: 10px; padding: 12px; margin-bottom: 14px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="background: #0284c7; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white;">
                    {user.get('full_name', 'O')[0]}
                </div>
                <div>
                    <div style="font-size: 0.88rem; font-weight: 600; color: #f8fafc; line-height: 1.2;">
                        {user.get('full_name', 'Officer')}
                    </div>
                    <div style="font-size: 0.75rem; color: #38bdf8;">
                        {user.get('role', 'Investigator')}
                    </div>
                </div>
            </div>
            <div style="margin-top: 8px; border-top: 1px solid #334155; padding-top: 6px; font-size: 0.72rem; color: #94a3b8; display: flex; justify-content: space-between;">
                <span>ID: <code>{user.get('username', '')}</code></span>
                <span style="color: {'#34d399' if status['is_connected'] else '#fbbf24'};">
                    {'● Mongo: Connected' if status['is_connected'] else '○ Mongo: Offline'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.sidebar.button("🚪 Sign Out", key="sidebar_logout_btn", use_container_width=True):
            logout_user()
