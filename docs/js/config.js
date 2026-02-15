const CONFIG = {
  development: {
    API_BASE: "http://localhost:8000",
  },
  production: {
    API_BASE: "https://matrix-tools.onrender.com",
  },
};

const isLocal =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1";

const ENV = isLocal ? "development" : "production";
const API_BASE = CONFIG[ENV].API_BASE;

window.API_BASE = API_BASE;
window.APP_ENV = ENV;
