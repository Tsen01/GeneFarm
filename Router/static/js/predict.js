// 顯示 Toast 訊息
function showToast(msg, type="success") {
  const toast = document.createElement('div');
  toast.className = 'toast ' + type;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => { toast.classList.add('hide'); setTimeout(() => toast.remove(), 500); }, 3000);
}

// 渲染預測結果
function renderPredictionResult(result) {
  let html = `<p class="w3-large"><b>羊隻編號：</b> ${result.earnum}</p>`;

  // 預測表格
  for (const [modelName, preds] of Object.entries(result.predictions)) {
    html += `<h4>${modelName} 預測結果</h4>
      <table class="w3-table-all w3-large">
        <thead><tr><th>日期</th><th>天數</th><th>體重 (kg)</th></tr></thead><tbody>`;
    preds.forEach(p => {
      html += `<tr><td>${p.date}</td><td>${p.days}</td><td>${p.weight.toFixed(2)}</td></tr>`;
    });
    html += `</tbody></table>`;
  }

  // 模型評估指標
  if (result.metrics) {
    html += `<h4>模型評估指標</h4>
      <table class="w3-table-all w3-large">
      <thead><tr><th>模型</th><th>RMSE</th><th>MAE</th><th>MSE</th></tr></thead><tbody>`;
    for (const [modelName, metric] of Object.entries(result.metrics)) {
      const rmse = metric?.rmse ?? metric?.RMSE;
      const mae  = metric?.mae  ?? metric?.MAE;
      const mse  = metric?.mse  ?? metric?.MSE;
      html += `<tr><td>${modelName}</td><td>${rmse ? rmse.toFixed(3) : "-"}</td><td>${mae ? mae.toFixed(3) : "-"}</td><td>${mse ? mse.toFixed(3) : "-"}</td></tr>`;
    }
    html += `</tbody></table><br>`;
  }

    // 成長曲線
    if (result.predictions) {
        html += `<div id="growthChart" style="width:100%; height:400px;"></div>`;
        setTimeout(() => {
        const seriesData = [
            {
            name: "實際體重",
            data: (result.actual || []).map(a => [Date.parse(a.date), a.weight]),
            color: "#3498db",
            marker: { symbol: "circle" }
            },
            {
            name: "Bi-LSTM + Attention",
            data: result.predictions["Bi-LSTM_Attention"].map(p => [Date.parse(p.date), p.weight]),
            color: "#e74c3c",
            marker: { symbol: "diamond" }
            },
            {
            name: "Bi-LSTM",
            data: result.predictions["Bi-LSTM_no_Attention"].map(p => [Date.parse(p.date), p.weight]),
            color: "#9b59b6",
            marker: { symbol: "triangle" }
            },
            {
            name: "Linear Regression",
            data: result.predictions["Linear_Regression"].map(p => [Date.parse(p.date), p.weight]),
            color: "#2ecc71",
            marker: { symbol: "square" }
            }
        ];

        Highcharts.chart('growthChart', {
            chart: { type: 'line' },
            title: { text: '羊隻成長預測曲線' },
            xAxis: { type: 'datetime', title: { text: '時間' } },
            yAxis: { title: { text: '體重 (kg)' } },
            tooltip: { shared: true, xDateFormat: '%Y-%m-%d', valueDecimals: 2, valueSuffix: ' kg' },
            legend: { enabled: true },
            series: seriesData
        });
        }, 100);
    }

  document.getElementById("resultArea").innerHTML = html;
  showToast("預測完成");
}

// 表單送出
document.getElementById("predictForm").onsubmit = async function(e) {
  e.preventDefault();
  const inputs = Array.from(e.target.querySelectorAll("input[name^='input_data']"));
  const input_data = inputs.map(i => i.value);
  const token = localStorage.getItem("userToken");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": `Bearer ${token}` },
      body: JSON.stringify({ model_type: "lstm", input_data })
    });
    if (!res.ok) throw new Error("預測失敗");
    const result = await res.json();
    renderPredictionResult(result);
  } catch (err) {
    showToast("錯誤：" + err.message, "fail");
  }
};

// CSV 選羊 Modal
document.addEventListener("DOMContentLoaded", () => {
  const modal = document.getElementById("sheepModal");
  const sheepSelect = document.getElementById("sheepSelect");

  document.getElementById("selectSheepBtn").onclick = async () => {
    try {
      const res = await fetch("/get_sheep_list");
      const sheepList = await res.json();
      sheepSelect.innerHTML = sheepList.map(s => `<option value="${s}">${s}</option>`).join("");
      modal.style.display = "block";
    } catch {
      showToast("無法載入羊隻清單", "fail");
    }
  };

  document.getElementById("closeModal").onclick = () => modal.style.display = "none";

  document.getElementById("loadSheepBtn").onclick = async () => {
    const sheepId = sheepSelect.value;
    if (!sheepId) return;
    const token = localStorage.getItem("userToken");
    try {
      const res = await fetch(`/get_sheep_data?earnum=${sheepId}`, {
        headers: { "Authorization": `Bearer ${token}` }
      });
      const input_data = await res.json();
      const predictRes = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": `Bearer ${token}` },
        body: JSON.stringify({ model_type: "lstm", input_data })
      });
      const result = await predictRes.json();
      renderPredictionResult(result);
      modal.style.display = "none";
    } catch {
      showToast("載入資料或預測失敗", "fail");
    }
  };
});