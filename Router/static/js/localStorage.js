// === 共用方法 ===
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

  // 根據角色決定跳轉與按鈕顯示
  if (data.role === "GeneticResearcher") {
    window.location.href = "/AnimalManager/gene";
  } else {
    window.location.href = "/AnimalManager/myfarm";
  }
}

// 更新 Navbar 顯示狀態
function updateNavbarVisibility() {
  const role = localStorage.getItem("userRole");

  if (role === "GeneticResearcher") {
    document.getElementById("growthBtn").style.display = "none";
    document.getElementById("role").style.display = "none";  // 隱藏「我的牧場」
  } else if (role === "Farmer") {
    document.getElementById("geneBtn").style.display = "none";
    document.getElementById("genePredictBtn").style.display = "none";
  }

  console.log("userName from localStorage:", localStorage.getItem('userName'));
}

// === 註冊 ===
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

  role = role.charAt(0).toUpperCase() + role.slice(1); // 確保首字大寫

  try {
    const res = await fetch("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, password, role })
    });

    if (res.ok) {
      alert("註冊成功，自動登入中...");

      // 自動登入
      const loginRes = await fetch("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
      });

      if (loginRes.ok) {
        const data = await loginRes.json();
        handleLoginSuccess(data); // 共用處理
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

// === 登入 ===
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
      handleLoginSuccess(data); // 共用處理
    } else {
      const errorText = await res.text();
      alert("登入失敗：" + errorText);
    }
  } catch (err) {
    console.error("登入錯誤", err);
    alert("登入時發生錯誤");
  }
}

// === 其他功能 ===
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
  openTab({ currentTarget: document.querySelector('.tablink:nth-child(2)') }, 'SignupForm');
}

// 頁面載入時自動執行
window.onload = updateNavbarVisibility;