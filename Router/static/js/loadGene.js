let currentPage = 1;
const rowsPerPage = 10;
let allData = [];            // 目前選擇的資料集資料
let currentFields = [];      // 資料欄位
let currentCollection = "";  // 當前選擇資料集

// 渲染表格
function renderTablePage(page) {
    const tbody = document.getElementById("geneTableBody");
    tbody.innerHTML = "";

    const header = document.getElementById("geneTableHead");
    header.innerHTML = "";

    if (!allData || allData.length === 0) {
        header.innerHTML = "<tr><th>尚未匯入基因資料集</th></tr>";
        tbody.innerHTML = "";
        document.getElementById("paginationControls").innerHTML = "";
        return;
    }

    // 表頭
    const trHead = document.createElement("tr");
    currentFields.forEach(field => {
        const th = document.createElement("th");
        th.textContent = field;
        trHead.appendChild(th);
    });
    header.appendChild(trHead);

    // 表格內容
    const start = (page - 1) * rowsPerPage;
    const end = Math.min(start + rowsPerPage, allData.length);

    for (let i = start; i < end; i++) {
        const row = document.createElement("tr");
        currentFields.forEach(field => {
            const td = document.createElement("td");
            td.textContent = allData[i][field] ?? "";
            row.appendChild(td);
        });
        tbody.appendChild(row);
    }

    renderPaginationControls();
}

// 分頁控制
function renderPaginationControls() {
    const div = document.getElementById("paginationControls");
    div.innerHTML = "";

    if (!allData || allData.length === 0) return;

    const totalPages = Math.ceil(allData.length / rowsPerPage);
    if (totalPages <= 1) return;

    const makeBtn = (text, fn, disabled) => {
        const btn = document.createElement("button");
        btn.textContent = text;
        btn.disabled = disabled;
        btn.className = "w3-button w3-border w3-small w3-margin";
        btn.onclick = fn;
        return btn;
    };

    div.appendChild(makeBtn("⏮️ 第一頁", () => { currentPage = 1; renderTablePage(currentPage); }, currentPage === 1));
    div.appendChild(makeBtn("◀️ 上一頁", () => { currentPage--; renderTablePage(currentPage); }, currentPage === 1));

    const select = document.createElement("select");
    for (let i = 1; i <= totalPages; i++) {
        const option = document.createElement("option");
        option.value = i;
        option.textContent = `第 ${i} 頁`;
        if (i === currentPage) option.selected = true;
        select.appendChild(option);
    }
    select.onchange = () => {
        currentPage = Number(select.value);
        renderTablePage(currentPage);
    };
    div.appendChild(select);

    div.appendChild(makeBtn("下一頁 ▶️", () => { currentPage++; renderTablePage(currentPage); }, currentPage === totalPages));
    div.appendChild(makeBtn("最後一頁 ⏭️", () => { currentPage = totalPages; renderTablePage(currentPage); }, currentPage === totalPages));
}

// 載入某個資料集
async function loadDataset(datasetName) {
    const token = localStorage.getItem("userToken");
    if (!token) return;

    if (!datasetName) {
        allData = [];
        currentFields = [];
        currentCollection = "";
        renderTablePage(1);
        return;
    }

    try {
        const res = await fetch(`/gene/data/${datasetName}`, {
            headers: { "Authorization": `Bearer ${token}` }
        });
        const result = await res.json();
        allData = result.data || [];
        currentFields = allData.length > 0 ? Object.keys(allData[0]).filter(k => k !== "user_id") : [];
        currentCollection = datasetName;
        currentPage = 1;
        renderTablePage(currentPage);
    } catch (err) {
        console.error("載入資料集失敗：", err);
        allData = [];
        currentFields = [];
        renderTablePage(1);
    }
}


// 匯入基因資料集
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return alert("請選擇檔案");
    
    const token = localStorage.getItem("userToken");
    if (!token) return alert("請先登入");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/gene/upload", {
            method: "POST",
            body: formData,
            headers: {
                "Authorization": `Bearer ${token}`
            }
        });

        const result = await res.json(); // 統一使用 result
        if (res.ok) {
            alert(`匯入成功！已新增 ${result.insertedCount} 筆資料`);
            await refreshDatasetSelect(result.collection);
        } else {
            alert("匯入失敗：" + (result.detail || JSON.stringify(result)));
        }
    } catch (err) {
        console.error("匯入錯誤", err);
        alert("匯入錯誤：" + err.message);
    }
}


// 刷新資料集選單
async function refreshDatasetSelect(loadDatasetName = null) {
    const token = localStorage.getItem("userToken");
    if (!token) return;

    try {
        const res = await fetch("/gene/collections", {
            headers: { "Authorization": `Bearer ${token}` }
        });
        const collections = await res.json();
        const select = document.getElementById("datasetSelect");
        select.innerHTML = "";

        if (!collections || collections.length === 0) {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = "尚未匯入基因資料集";
            select.appendChild(option);
            loadDataset(null);
        } else {
            collections.forEach(name => {
                const option = document.createElement("option");
                option.value = name;
                option.textContent = name;
                select.appendChild(option);
            });

            const datasetToLoad = loadDatasetName || collections[0];
            select.value = datasetToLoad;
            loadDataset(datasetToLoad);
        }

        select.onchange = () => loadDataset(select.value);
    } catch (err) {
        console.error("取得資料集列表失敗：", err);
    }
}

// 刪除資料集
async function deleteCollection() {
    const select = document.getElementById("datasetSelect");
    const collectionName = select.value;

    if (!collectionName) return alert("請先選擇資料集");
    if (!confirm(`確定要刪除資料集「${collectionName}」嗎？此操作無法復原！`)) return;

    try {
        const token = localStorage.getItem("userToken");
        const res = await fetch(`/gene/delete_collection/${collectionName}`, {
            method: "DELETE",
            headers: { "Authorization": `Bearer ${token}` }
        });

        if (!res.ok) {
            const data = await res.json();
            throw new Error(data.detail || "刪除失敗");
        }

        alert(`資料集「${collectionName}」已刪除`);
        await refreshDatasetSelect();  // 刪除後刷新選單，不需要 reload

    } catch (err) {
        console.error(err);
        alert("刪除資料集失敗：" + err.message);
    }
}

// 編輯資料集
function editCollection() {
    const select = document.getElementById("datasetSelect");
    const collectionName = select.value;
    if (!collectionName) return alert("請先選擇資料集");

    window.location.href = `/AnimalManager/EditGene/${collectionName}`;
}

// DOM 初始化
document.addEventListener("DOMContentLoaded", () => {
    refreshDatasetSelect();
});