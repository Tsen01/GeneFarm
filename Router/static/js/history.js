let allData = [];
let currentPage = 1;
const rowsPerPage = 10;

// 格式化時間 -> 台灣時間 (YYYY/MM/DD HH:mm)
function formatDateTime(value) {
    if (!value) return "—";

    // 如果是 MongoDB 帶 6 位數微秒的格式，把它截成 3 位數毫秒
    if (typeof value === "string" && value.match(/\.\d{6}$/)) {
        value = value.replace(/(\.\d{3})\d{3}$/, "$1");
    }

    const d = new Date(value);
    if (isNaN(d.getTime())) return value; // 如果不是合法時間就原樣輸出

    // 轉換成台灣時區 (Asia/Taipei)
    return d.toLocaleString("zh-TW", {
        timeZone: "Asia/Taipei",
        year: "numeric",
        month: "numeric",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false
    }).replace(/\//g, "/"); // 保持 YYYY/MM/DD 格式
}

// 初始化歷史資料
async function initHistory() {
    const token = localStorage.getItem("userToken");

    if (!token) {
        document.getElementById("result").textContent = "尚未登入";
        return;
    }

    try {
        const res = await fetch("/get_personal_data", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            },
            body: JSON.stringify({})
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            console.log("後端錯誤回傳:", errData);
            document.getElementById("result").textContent = errData.detail || "抓取歷史資料失敗";
            return;
        }

        const jsonData = await res.json();
        allData = jsonData.data || [];
        console.log("抓到的歷史資料:", allData);
        renderTablePage(currentPage);
        renderPaginationControls();

    } catch (err) {
        console.error(err);
        document.getElementById("result").textContent = "載入失敗";
    }
}

// 渲染分頁資料
function renderTablePage(page) {
    const container = document.getElementById("result");
    container.innerHTML = "";

    if (!allData || allData.length === 0) {
        container.textContent = "目前沒有歷史預測紀錄";
        return;
    }

    const start = (page - 1) * rowsPerPage;
    const end = Math.min(start + rowsPerPage, allData.length);

    // 過濾掉 user_id 欄位
    const fields = Object.keys(allData[0]).filter(f => f !== "user_id");

    const table = document.createElement("table");
    table.className = "w3-table w3-bordered w3-striped w3-hoverable";

    // 表頭
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    fields.forEach(f => {
        const th = document.createElement("th");
        th.textContent = f;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    // 表格內容
    const tbody = document.createElement("tbody");
    allData.slice(start, end).forEach(row => {
        const tr = document.createElement("tr");
        fields.forEach(f => {
            const td = document.createElement("td");
            let value = row[f] ?? "—";

            // 如果是時間字串就轉換成本地台灣時間
            if (typeof value === "string" && !isNaN(Date.parse(value))) {
                value = formatDateTime(value);
            }

            td.textContent = value;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    container.appendChild(table);
}

// 分頁控制
function renderPaginationControls() {
    const paginationDiv = document.getElementById("paginationControls");
    paginationDiv.innerHTML = "";
    const totalPages = Math.ceil(allData.length / rowsPerPage);
    if (totalPages <= 1) return;

    const createBtn = (text, onClick, disabled = false) => {
        const btn = document.createElement("button");
        btn.textContent = text;
        btn.className = "w3-button w3-border w3-small w3-margin";
        btn.disabled = disabled;
        btn.onclick = onClick;
        return btn;
    };

    paginationDiv.appendChild(createBtn("⏮️ 第一頁", () => { currentPage = 1; renderTablePage(currentPage); renderPaginationControls(); }, currentPage === 1));
    paginationDiv.appendChild(createBtn("◀️ 上一頁", () => { if (currentPage > 1) currentPage--; renderTablePage(currentPage); renderPaginationControls(); }, currentPage === 1));

    const select = document.createElement("select");
    select.className = "w3-border w3-round";
    for (let i = 1; i <= totalPages; i++) {
        const option = document.createElement("option");
        option.value = i;
        option.textContent = `第 ${i} 頁`;
        if (i === currentPage) option.selected = true;
        select.appendChild(option);
    }
    select.onchange = () => { currentPage = parseInt(select.value); renderTablePage(currentPage); renderPaginationControls(); };
    paginationDiv.appendChild(select);

    paginationDiv.appendChild(createBtn("下一頁 ▶️", () => { if (currentPage < totalPages) currentPage++; renderTablePage(currentPage); renderPaginationControls(); }, currentPage === totalPages));
    paginationDiv.appendChild(createBtn("最後一頁 ⏭️", () => { currentPage = totalPages; renderTablePage(currentPage); renderPaginationControls(); }, currentPage === totalPages));
}

// 使用 addEventListener 避免覆蓋 window.onload
window.addEventListener("load", initHistory);
