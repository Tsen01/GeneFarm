// 登出功能
function logoutUser() {
  if(confirm("確定要登出嗎？")) {
    localStorage.clear(); 
    window.location.href = "/AnimalManager"; // 回到首頁
  }
}

// UI 切換功能 (選單與 Tab)
function openNav(){
  var x = document.getElementById("navDemo");
  if (x.className.indexOf("w3-show") == -1) {
    x.className += " w3-show";
  } else {
    x.className = x.className.replace(" w3-show", "");
  }
}

function openTab(evt, tabName) {
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  tablinks = document.getElementsByClassName("tablink");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" w3-teal", "");
  }
  var targetTab = document.getElementById(tabName);
  if (targetTab) targetTab.style.display = "block";
  if (evt) evt.currentTarget.className += " w3-teal";
}

// 身分驗證與資料存取
function saveUserData(data) {
  localStorage.clear();
  localStorage.setItem("loggedIn", "true");
  localStorage.setItem("userName", data.username);
  localStorage.setItem("userRole", data.role);
  localStorage.setItem("userToken", data.token);
}

function handleLoginSuccess(data) {
  saveUserData(data);
  alert("登入成功！");
  const modal = document.getElementById("id01");
  if (modal) modal.style.display = "none";

  // 更新 UI 並跳轉
  updateNavbarVisibility();

  if (data.role === "GeneticResearcher") {
    window.location.href = "/AnimalManager/gene";
  } else {
    window.location.href = "/AnimalManager/myfarm";
  }
}

// 更新 Navbar (控制顯示)
function updateNavbarVisibility() {
  // 取得資料
  const role = localStorage.getItem("userRole");
  const userName = localStorage.getItem("userName");
  const isLoggedIn = localStorage.getItem("loggedIn") === "true";

  // 取得 DOM 元素
  const desktopAuthBtn = document.getElementById("authButtons");

  // 電腦版按鈕
  const desktopMyFarm = document.getElementById("role");
  const desktopGrowth = document.getElementById("growthBtn");
  const desktopGene = document.getElementById("geneBtn");
  const desktopGenePredict = document.getElementById("genePredictBtn");

  // 手機版按鈕
  const mobileMyFarm = document.getElementById("mobile_myfarm");
  const mobileGrowth = document.getElementById("mobile_growth");
  const mobileGene = document.getElementById("mobile_gene");
  const mobileGenePredict = document.getElementById("mobile_genePredict");

  // --- 處理登入/登出顯示 ---
  const roleLabel = (role === "GeneticResearcher") ? "基因研究員" : "牧場主";
  
  const loginHtml = `<button onclick="document.getElementById('id01').style.display='block'" class="w3-button w3-theme w3-hover-teal" title="登入/註冊">登入/註冊</button>`;
  
  const welcomeHtml = `
    <span class="w3-bar-item">歡迎 ${userName} (${roleLabel})</span>
    <button onclick="logoutUser()" class="w3-button w3-theme w3-hover-red">登出</button>
  `;

  if (isLoggedIn) {
      // 已登入時，替換成歡迎文字，並設定 display: block (顯示)
      if (desktopAuthBtn) {
          desktopAuthBtn.innerHTML = welcomeHtml;
          desktopAuthBtn.style.display = "block"; 
      }
  } else {
      // 未登入：顯示登入按鈕
      if (desktopAuthBtn) {
          desktopAuthBtn.innerHTML = loginHtml;
          desktopAuthBtn.style.display = "block";
      }
  }

  // --- 處理身分選單 ---
  if (!role) return; 

  if (role === "GeneticResearcher") {
    // 顯示基因
    if(desktopGene) desktopGene.style.display = "block";
    if(desktopGenePredict) desktopGenePredict.style.display = "block";
    if(mobileGene) mobileGene.style.display = "block";
    if(mobileGenePredict) mobileGenePredict.style.display = "block";
    
    // 隱藏牧場
    if(desktopMyFarm) desktopMyFarm.style.display = "none";
    if(desktopGrowth) desktopGrowth.style.display = "none";
    if(mobileMyFarm) mobileMyFarm.style.display = "none";
    if(mobileGrowth) mobileGrowth.style.display = "none";

  } else if (role === "Farmer") {
    // 顯示牧場
    if(desktopMyFarm) desktopMyFarm.style.display = "block";
    if(desktopGrowth) desktopGrowth.style.display = "block";
    if(mobileMyFarm) mobileMyFarm.style.display = "block";
    if(mobileGrowth) mobileGrowth.style.display = "block";

    // 隱藏基因
    if(desktopGene) desktopGene.style.display = "none";
    if(desktopGenePredict) desktopGenePredict.style.display = "none";
    if(mobileGene) mobileGene.style.display = "none";
    if(mobileGenePredict) mobileGenePredict.style.display = "none";
  }

  console.log("UI Updated. User:", userName);
}

// 註冊
async function registerUser(event) {
  event.preventDefault();
  let username = document.getElementById("signup_FarmName").value.trim();
  let email = document.getElementById("signup_mail").value;
  let password = document.getElementById("signup_pw").value;
  let role = document.getElementById("userRole").value;

  if (!username || !email || !password || !role) {
    alert("請填寫完整的註冊資訊");
    return;
  }
  role = role.charAt(0).toUpperCase() + role.slice(1);

  try {
    const res = await fetch("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, password, role })
    });

    if (res.ok) {
      alert("註冊成功，自動登入中...");
      const loginRes = await fetch("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
      });
      if (loginRes.ok) {
        const data = await loginRes.json();
        handleLoginSuccess(data);
      } else {
        alert("自動登入失敗，請手動登入");
        document.getElementById("id01").style.display = "block";
      }
    } else {
      const errorText = await res.text();
      alert("註冊失敗：" + errorText);
    }
  } catch (err) {
    console.error("註冊錯誤", err);
    alert("註冊失敗：伺服器錯誤");
  }
}

// 登入
async function loginUser(event) {
  event.preventDefault();
  const email = document.getElementById("login_mail").value;
  const password = document.getElementById("login_pw").value;

  if (!email || !password) {
    alert("請輸入 email 與密碼！");
    return;
  }
  try {
    const res = await fetch("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password })
    });
    if (res.ok) {
      const data = await res.json();
      handleLoginSuccess(data);
    } else {
      const errorText = await res.text();
      alert("登入失敗：" + errorText);
    }
  } catch (err) {
    console.error("登入錯誤", err);
    alert("登入時發生錯誤");
  }
}

// 其他功能
function checkLoginAndRedirect(targetUrl) {
  const loggedIn = localStorage.getItem("loggedIn") === "true";
  if (loggedIn) {
    window.location.href = targetUrl;
  } else {
    const loginModal = document.getElementById("id01");
    if (loginModal) {
      loginModal.style.display = "block";
    } else {
      alert("請先登入或註冊以瀏覽資料");
    }
  }
}

function openSignupTab() {
  document.getElementById("id01").style.display = "block";
  const signupTabLink = document.querySelectorAll('.tablink')[1];
  if (signupTabLink) {
    openTab({ currentTarget: signupTabLink }, 'SignupForm');
  }
}

// 頁面載入時自動執行
window.onload = function() {
    updateNavbarVisibility();
};