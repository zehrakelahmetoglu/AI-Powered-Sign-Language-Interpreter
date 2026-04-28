import streamlit as st
import streamlit.components.v1 as components
import time
from datetime import datetime
st.set_page_config(layout="wide", page_title="İşitme Engelli Hasta İletişim Sistemi", page_icon="💡", initial_sidebar_state="expanded")

# Session State
if "cumle" not in st.session_state:
    st.session_state.cumle = ""
if "kelimeler" not in st.session_state:
    st.session_state.kelimeler = []

# 2. Canlı algılama efekti
if "last_len" not in st.session_state:
    st.session_state.last_len = 0

if len(st.session_state.kelimeler) > st.session_state.last_len:
    st.toast("Yeni kelime algılandı!", icon="🧠")
st.session_state.last_len = len(st.session_state.kelimeler)

# CSS Stilleri
st.markdown("""
<style>
    /* Streamlit Varsayılan UI Öğelerini Gizleme / Resetleme */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden !important;}
    footer {visibility: hidden;}
    
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* Genel Arka Plan ve Metin */
    .stApp {
        background-color: #ffffff;
        color: #212529;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Özel Üst Menü Barı (Header) PRO YENİLEME */
    .custom-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 30px;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-bottom: 1px solid rgba(0,0,0,0.05);
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    }
    
    .header-left {
        display: flex;
        align-items: center;
        gap: 15px;
        font-weight: 700;
        color: #1a1b26;
        font-size: 20px;
        letter-spacing: -0.5px;
    }
    
    .logo-box {
        background: linear-gradient(135deg, #198754 0%, #0f5132 100%);
        color: white;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 20px;
        box-shadow: 0 4px 10px rgba(25, 135, 84, 0.3);
    }
    
    .header-right {
        color: #333333;
        font-size: 14px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
        background: white;
        padding: 8px 16px;
        border-radius: 30px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02), 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .green-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px rgba(220, 53, 69, 0.6);
    }

    /* Sağ Panel "Metin Dökümü" Başlıkları */
    .doc-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #333333;
        padding-bottom: 15px;
        margin-bottom: 10px;
        border-bottom: 1px solid #f0f0f0;
    }

    .doc-title {
        font-size: 15px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .word-count {
        font-size: 12px;
        color: #333333;
    }

    /* Buton Stilleri */
    button[kind="secondary"] {
        background-color: transparent !important;
        color: #6c757d !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        justify-content: center !important;
        font-weight: 500 !important;
    }
    button[kind="secondary"]:hover {
        color: #198754 !important;
    }

    button[kind="primary"] {
        background-color: #198754 !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
        width: 100% !important;
        font-weight: 500 !important;
        transition: 0.2s;
    }
    button[kind="primary"]:hover {
        background-color: #146c43 !important;
    }
</style>
""", unsafe_allow_html=True)

# Özel Üst Menü (Startup Dashboard Style)
tarih = datetime.now().strftime("%d %B %Y")

st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; padding:15px 20px; background: linear-gradient(135deg, #f8f9fa, #ffffff); border-radius:12px; margin-bottom:20px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.05);">
<div style="display:flex; align-items:center; gap:15px;">
<div style="background: linear-gradient(135deg, #198754 0%, #0f5132 100%); color:white; width:42px; height:42px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:20px; box-shadow: 0 4px 10px rgba(25, 135, 84, 0.3);">💡</div>
<div>
<div style="font-size:16px; font-weight:700; color:#1a1b26; letter-spacing:-0.5px;">Hasta İletişim Paneli</div>
</div>
</div>
<div style="display:flex; gap:30px; text-align:center;">
<div>
<div style="font-size:12px; color:#6c757d;">Algılanan</div>
<div style="font-weight:700; color:#1a1b26; font-size:18px;">{len(st.session_state.kelimeler)}</div>
</div>
<div style="width: 1px; background-color: #e9ecef; margin-top:5px; margin-bottom:5px;"></div>
<div>
<div style="font-size:12px; color:#6c757d;">Toplam Kelime</div>
<div style="font-weight:700; color:#1a1b26; font-size:18px;">{len(st.session_state.cumle.split())}</div>
</div>
</div>
<div style="display:flex; flex-direction:column; align-items:flex-end; gap:5px;">
<div style="font-size:12px; color:#999; font-weight:500;">{tarih}</div>
<div style="display:flex; align-items:center; gap:8px;">
<div id="topStatusDot" style="width:10px; height:10px; background:#dc3545; border-radius:50%; box-shadow: 0 0 8px rgba(220, 53, 69, 0.6); transition: 0.3s;"></div>
<div id="topStatusText" style="font-size:14px; font-weight:bold; color:#dc3545; transition: 0.3s;">Sistem Pasif</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# Ekranı böl
col1, col2 = st.columns([1.1, 1])

# 🎥 SOL: Kamera Oynatıcı
with col1:
    camera_html = """
    <div style="background-color: #0f0f0f; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; font-family: 'Segoe UI', sans-serif;">
        <!-- Video Alanı -->
        <div style="position: relative; width: 100%; height: 500px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <video id="videoElement" autoplay playsinline style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; display: none;"></video>
            
            <!-- 4. Kamera UI -> CANLI Overlay -->
            <div id="liveOverlay" style="position:absolute; top:15px; left:15px; background: rgba(220,53,69,0.9); color:white; padding:4px 10px; border-radius:4px; font-size:12px; font-weight:bold; display:none; z-index: 20; letter-spacing: 1px;">
                🔴 CANLI
            </div>

            <div id="placeholder" style="text-align: center; color: #555555; z-index: 10;">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="width: 48px; height: 48px; margin-bottom: 10px;">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                    <line x1="2" y1="2" x2="22" y2="22" stroke="currentColor" stroke-width="2"/>
                </svg>
                <div style="font-size: 14px;">Kamera erişimi bekleniyor</div>
                <div id="statusPill" style="margin-top: 15px; padding: 4px 16px; border-radius: 20px; border: 1px solid #333333; font-size: 13px; display: inline-block;">— Bekleniyor —</div>
            </div>
            
            <!-- Basit Progress Bar Tasarımı -->
            <div style="position: absolute; bottom: 15px; left: 20px; right: 20px; display: flex; align-items: center; gap: 15px; z-index: 20;">
                <button id="startBtn" style="background-color: #198754; color: white; border: none; padding: 6px 16px; border-radius: 4px; font-size: 13px; font-weight: 500; cursor: pointer;">Kamerayı Başlat</button>

            </div>
        </div>
        


        <script>
            const video = document.getElementById('videoElement');
            const startBtn = document.getElementById('startBtn');
            const placeholder = document.getElementById('placeholder');
            const statusPill = document.getElementById('statusPill');
            const liveOverlay = document.getElementById('liveOverlay');
            let stream = null;

            function updateParentUI(isActive) {
                try {
                    const parentDoc = window.parent.document;
                    
                    const topText = parentDoc.getElementById('topStatusText');
                    const topDot = parentDoc.getElementById('topStatusDot');
                    if (topText && topDot) {
                        if (isActive) {
                            topText.innerText = "Sistem Aktif";
                            topText.style.color = "#198754";
                            topDot.style.backgroundColor = "#198754";
                            topDot.style.boxShadow = "0 0 8px rgba(25, 135, 84, 0.6)";
                        } else {
                            topText.innerText = "Sistem Pasif";
                            topText.style.color = "#dc3545";
                            topDot.style.backgroundColor = "#dc3545";
                            topDot.style.boxShadow = "0 0 8px rgba(220, 53, 69, 0.6)";
                        }
                    }

                    const styleId = "camera-overlay-style";
                    let styleEl = parentDoc.getElementById(styleId);
                    if (!styleEl) {
                        styleEl = parentDoc.createElement('style');
                        styleEl.id = styleId;
                        parentDoc.head.appendChild(styleEl);
                    }
                    if (isActive) {
                        styleEl.innerHTML = "";
                    } else {
                        styleEl.innerHTML = `
                            [data-testid="column"]:nth-child(2),
                            [data-testid="stColumn"]:nth-child(2),
                            button[kind="primary"], 
                            button[kind="secondary"],
                            #docArea {
                                opacity: 0.4 !important;
                                pointer-events: none !important;
                                filter: grayscale(50%) !important;
                                transition: all 0.3s ease-in-out;
                            }
                        `;
                    }
                } catch(e) {
                    console.error("DOM Error: ", e);
                }
            }

            // Başlangıçta sağ paneli ve statüyü pasif yap
            updateParentUI(false);

            startBtn.addEventListener('click', async () => {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    video.srcObject = null;
                    stream = null;
                    video.style.display = 'none';
                    liveOverlay.style.display = 'none';
                    placeholder.style.display = 'block';
                    startBtn.textContent = 'Kamerayı Başlat';
                    startBtn.style.backgroundColor = '#198754';
                    statusPill.textContent = '— Bekleniyor —';
                    updateParentUI(false);
                } else {
                    try {
                        stream = await navigator.mediaDevices.getUserMedia({ video: true });
                        video.srcObject = stream;
                        video.style.display = 'block';
                        liveOverlay.style.display = 'block';
                        placeholder.style.display = 'none';
                        startBtn.textContent = 'Durdur';
                        startBtn.style.backgroundColor = '#dc3545';
                        updateParentUI(true);
                    } catch (err) {
                        alert("Kamera erişimi reddedildi veya bulunamadı.");
                    }
                }
            });
        </script>
    </div>
    """
    components.html(camera_html, height=550)

# 📝 SAĞ: Belge Dökümü
with col2:
    # Başlık
    st.markdown("""
    <div class="doc-header">
        <div class="doc-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line></svg>
            Metin Dökümü
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. & 7. Boş Alan Yönetimi, Glassmorphism ve Typing Effect
    doc_placeholder = st.empty()
    
    def render_doc(content):
        return f"""
        <div id="docArea" style="
            height: 380px;
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 10px;
            padding: 20px;
            font-size: 16px;
            color: #212529;
            line-height: 1.6;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
            overflow-y: auto;
        ">
        {content}
        </div>
        """

    if "last_cumle" not in st.session_state:
        st.session_state.last_cumle = ""

    if st.session_state.cumle != st.session_state.last_cumle and st.session_state.cumle != "":
        temp_text = ""
        for char in st.session_state.cumle:
            temp_text += char
            doc_placeholder.markdown(render_doc(temp_text), unsafe_allow_html=True)
            time.sleep(0.01)
        st.session_state.last_cumle = st.session_state.cumle
    else:
        if st.session_state.cumle:
            doc_placeholder.markdown(render_doc(st.session_state.cumle), unsafe_allow_html=True)
        else:
            doc_placeholder.markdown(render_doc("<span style='color:#aaa;'>Metin burada görünecek...</span>"), unsafe_allow_html=True)

    # Alt Yazılar
    st.markdown("<div style='font-size: 13px; color: #333333; margin-top: 15px; margin-bottom: 5px;'>Algılanan kelimeler (gönderilmemiş):</div>", unsafe_allow_html=True)
    
    # 5. Kelimeleri Badge Yap (ŞART)
    if st.session_state.kelimeler:
        badges = "".join([
            f"<span style='background:#e7f1ff; color:#0d6efd; padding:6px 12px; border-radius:20px; margin:4px; display:inline-block; font-size:14px; font-weight:500;'>{k}</span>"
            for k in st.session_state.kelimeler
        ])
        st.markdown(badges, unsafe_allow_html=True)
    else:
        st.markdown("<div style='color: #6c757d; font-size: 14px;'>Henüz kelime yok...</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Butonlar
    # 9. En Önemli Upgrade ("Cümleye Ekle" butonu)
    btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1.5, 3])
    
    with btn_col1:
        if st.button("➕ Cümleye Ekle", type="secondary"):
            if st.session_state.kelimeler:
                st.session_state.cumle += " ".join(st.session_state.kelimeler) + " "
                st.session_state.kelimeler = []
                st.rerun()

    with btn_col2:
        if st.button("🗑️ Temizle", type="secondary"):
            st.session_state.cumle = ""
            st.session_state.kelimeler = []
            st.rerun()

    with btn_col3:
        # 6. Buton UX (Gönder butonu upgrade)
        send_clicked = st.button("Gönder ↗", type="primary")

    if send_clicked:
        if st.session_state.cumle.strip():
            st.success("Doktora iletildi ✅")
            st.balloons()
            st.markdown(f"""
            <div style="
            margin-top:15px;
            padding:15px;
            background:#f8f9fa;
            border-left:4px solid #198754;
            border-radius:6px;
            ">
            🧑‍⚕️ <b>Doktor ekranına düşen mesaj:</b><br><br>
            {st.session_state.cumle}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
            margin-top:15px;
            padding:12px 15px;
            background:#f8f9fa;
            border-left:4px solid #6c757d;
            border-radius:6px;
            color: #212529;
            font-size: 15px;
            ">
            ℹ️ Gönderilecek bir metin yok. Lütfen önce kelime ekleyin.
            </div>
            """, unsafe_allow_html=True)