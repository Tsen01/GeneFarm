window.addEventListener('DOMContentLoaded', function () {
  const growthBtn = document.getElementById("growthBtn");
  const geneBtn = document.getElementById("geneBtn");
  const genePredictBtn = document.getElementById("genePredictBtn");
  const authButtons = document.getElementById('authButtons');

  // 先讀取登入狀態
  const userName = localStorage.getItem('userName');
  const role = localStorage.getItem('userRole');
  const loggedIn = localStorage.getItem('loggedIn') === "true";
  const roleLink = document.getElementById("role");

  if (!loggedIn || !authButtons) return; // 沒登入就什麼都不顯示

  const roleLabel = role === "GeneticResearcher" ? "基因研究員" : "牧場主";

  // 更新右上角登入狀態
  authButtons.innerHTML = `
    <span class="w3-bar-item w3-margin-right">歡迎 ${userName}（${roleLabel}）</span>
    <button onclick="logout()" class="w3-button w3-theme w3-hover-red">登出</button>
  `;

  // 顯示對應的按鈕
  if (role === "GeneticResearcher" && geneBtn) {
    geneBtn.style.display = "inline-block";
    genePredictBtn.style.display = "inline-block";
  } else if (growthBtn) {
    roleLink.style.display = "inline-block";
    growthBtn.style.display = "inline-block";
  }
});

function logout() {
  localStorage.clear();
  window.location.href = "/AnimalManager"; // 重定向到首頁
}
