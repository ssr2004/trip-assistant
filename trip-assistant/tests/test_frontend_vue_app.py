"""
Vue前端工程测试
"""
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_DIR / "frontend"


def test_frontend_vite_entry_exists():
    """前端已升级为Vite入口"""
    index_html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    assert '<div id="app"></div>' in index_html
    assert 'src="/src/main.js"' in index_html


def test_frontend_package_has_vue_build_scripts():
    """前端package包含Vue和构建脚本"""
    package_json = (FRONTEND_DIR / "package.json").read_text(encoding="utf-8")

    assert '"vue"' in package_json
    assert '"vite"' in package_json
    assert '"@lucide/vue"' in package_json
    assert '"build": "vite build"' in package_json


def test_frontend_calls_chat_and_external_status_apis():
    """前端调用聊天和外部状态接口"""
    api_js = (FRONTEND_DIR / "src" / "api.js").read_text(encoding="utf-8")
    app_vue = (FRONTEND_DIR / "src" / "App.vue").read_text(encoding="utf-8")

    assert 'fetch("/api/chat"' in api_js
    assert 'fetch("/api/external/status")' in api_js
    assert "travelMindSessionId" in app_vue
    assert "newSession" in app_vue
    assert "data.artifacts" in app_vue


def test_frontend_components_are_split_by_responsibility():
    """前端已拆分为聊天、状态和结构化卡片组件"""
    app_vue = (FRONTEND_DIR / "src" / "App.vue").read_text(encoding="utf-8")
    components_dir = FRONTEND_DIR / "src" / "components"

    assert (components_dir / "ChatPanel.vue").exists()
    assert (components_dir / "StatusPanel.vue").exists()
    assert (components_dir / "ArtifactCards.vue").exists()
    assert "import ChatPanel" in app_vue
    assert "import StatusPanel" in app_vue
    assert "<ChatPanel" in app_vue
    assert "<StatusPanel" in app_vue


def test_frontend_renders_structured_artifacts():
    """前端包含行程、天气、路线和景点结构化展示"""
    artifact_vue = (FRONTEND_DIR / "src" / "components" / "ArtifactCards.vue").read_text(encoding="utf-8")

    assert "artifacts.itinerary" in artifact_vue
    assert "artifacts.weather" in artifact_vue
    assert "artifacts.weather_adjustment" in artifact_vue
    assert "artifacts.route" in artifact_vue
    assert "artifacts.attractions" in artifact_vue


def test_frontend_avoids_raw_html_rendering():
    """前端不使用原始HTML渲染Agent回复"""
    source_files = [
        FRONTEND_DIR / "src" / "App.vue",
        FRONTEND_DIR / "src" / "components" / "ChatPanel.vue",
        FRONTEND_DIR / "src" / "components" / "ArtifactCards.vue",
        FRONTEND_DIR / "src" / "components" / "StatusPanel.vue",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in source_files)

    assert "v-html" not in combined
    assert "innerHTML" not in combined
    assert "{{ message.content }}" in combined
