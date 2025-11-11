// 打開忘記密碼視窗（你可改成你的觸發事件）
function openForgotPassword() {
    document.getElementById("id03").style.display = "block";
}
function closeModal() {
    document.getElementById("id03").style.display = "none";
    resetModal();
}

let userEmail = "";

async function sendCode() {
    userEmail = document.getElementById("resetEmail").value.trim();
    if (!userEmail) {
    alert("請輸入Email");
    return;
    }

    try {
    let res = await fetch("http://localhost:8000/send-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: userEmail }),
    });
    let data = await res.json();
    alert(data.message);
    document.getElementById("verifySection").style.display = "block";
    } catch (err) {
    alert("發送驗證碼失敗");
    console.error(err);
    }
}

async function verifyCode() {
    let code = document.getElementById("codeInput").value.trim();
    if (!code) {
    alert("請輸入驗證碼");
    return;
    }

    try {
    let res = await fetch("http://localhost:8000/verify-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: userEmail, code: code }),
    });
    if (!res.ok) {
        const errData = await res.json();
        document.getElementById("codeError").style.display = "inline";
        document.getElementById("codeError").textContent = errData.detail || "驗證碼錯誤";
        return;
    }
    document.getElementById("codeError").style.display = "none";
    alert("驗證成功，請設定新密碼");
    document.getElementById("resetForm").style.display = "block";
    } catch (err) {
    alert("驗證失敗");
    console.error(err);
    }
}

document.getElementById("resetForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    const pw = document.getElementById("newPw").value.trim();
    const confirmPw = document.getElementById("confirmPw").value.trim();

    if (!pw || !confirmPw) {
    alert("請輸入新的密碼");
    return;
    }
    if (pw !== confirmPw) {
    alert("密碼不一致，請重新輸入新密碼");
    return;
    }

    try {
    let res = await fetch("http://localhost:8000/reset-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: userEmail, password: pw }),
    });
    let data = await res.json();
    alert(data.message + "，請重新登入");
    closeModal();
    } catch (err) {
    alert("重設密碼失敗");
    console.error(err);
    }
});

// 重置 Modal 狀態
function resetModal() {
    document.getElementById("resetEmail").value = "";
    document.getElementById("codeInput").value = "";
    document.getElementById("newPw").value = "";
    document.getElementById("confirmPw").value = "";
    document.getElementById("verifySection").style.display = "none";
    document.getElementById("resetForm").style.display = "none";
    document.getElementById("codeError").style.display = "none";
    userEmail = "";
}