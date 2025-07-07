document.addEventListener('DOMContentLoaded', function() {
  let csvHeaders = [];

  // Show target column dropdown after file selection
  const fileInput = document.getElementById('csvFile');
  fileInput.onchange = function(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function(evt) {
      const firstLine = evt.target.result.split('\n')[0];
      csvHeaders = firstLine.replace('\r', '').split(',');
      // Create dropdown
      let dropdown = '<label for="targetColumn" class="form-label mt-2">Select Target Column:</label>';
      dropdown += '<select id="targetColumn" class="form-select mb-2">';
      csvHeaders.forEach(col => {
        dropdown += `<option value="${col}">${col}</option>`;
      });
      dropdown += '</select>';
      document.getElementById('targetColumnDropdown').innerHTML = dropdown;
    };
    reader.readAsText(file);
  };

  document.getElementById('uploadBtn').onclick = async function() {
    const statusDiv = document.getElementById('uploadStatus');
    if (!fileInput.files.length) {
      statusDiv.innerText = 'Please select a CSV file.';
      return;
    }
    const targetColumn = document.getElementById('targetColumn') ? document.getElementById('targetColumn').value : null;
    if (!targetColumn) {
      statusDiv.innerText = 'Please select a target column.';
      return;
    }
    const modelType = document.getElementById('modelSelect') ? document.getElementById('modelSelect').value : 'random_forest';
    statusDiv.innerText = 'Uploading...';
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('target_column', targetColumn);
    formData.append('model_type', modelType);
    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      if (!response.ok) {
        statusDiv.innerText = 'Upload failed. Server error.';
        return;
      }
      const data = await response.json();
      // Format metrics
      let metricsHtml = `
        <div class="card p-3 mb-3">
          <h5 class="mb-3 text-success">Upload successful!</h5>
          <div class="row g-3">
            <div class="col-6 col-md-4"><b>Accuracy:</b> ${data.accuracy}</div>
            <div class="col-6 col-md-4"><b>Precision:</b> ${data.precision}</div>
            <div class="col-6 col-md-4"><b>Recall:</b> ${data.recall}</div>
            <div class="col-6 col-md-4"><b>F1 Score:</b> ${data.f1}</div>
            <div class="col-6 col-md-4"><b>ROC AUC:</b> ${data.roc_auc !== null ? data.roc_auc : 'N/A'}</div>
          </div>
        </div>
      `;
      // Confusion matrix as table
      if (data.confusion_matrix && data.confusion_matrix.length > 0) {
        metricsHtml += '<div class="mb-2"><b>Confusion Matrix:</b></div>';
        metricsHtml += '<div style="overflow-x:auto;"><table class="table table-bordered table-sm confusion-matrix-table"><tbody>';
        data.confusion_matrix.forEach(row => {
          metricsHtml += '<tr>' + row.map(cell => `<td>${cell}</td>`).join('') + '</tr>';
        });
        metricsHtml += '</tbody></table></div>';
      }
      // Visualizations
      let vizHtml = '';
      if (data.confusion_matrix_img) {
        vizHtml += `<div class="mb-3"><b>Confusion Matrix (Plot):</b><br><img src="${data.confusion_matrix_img}" alt="Confusion Matrix" class="img-fluid" style="max-width:400px;"></div>`;
      }
      if (data.roc_curve_img) {
        vizHtml += `<div class="mb-3"><b>ROC Curve:</b><br><img src="${data.roc_curve_img}" alt="ROC Curve" class="img-fluid" style="max-width:400px;"></div>`;
      }
      if (data.feature_importance_img) {
        vizHtml += `<div class="mb-3"><b>Feature Importance:</b><br><img src="${data.feature_importance_img}" alt="Feature Importance" class="img-fluid" style="max-width:400px;"></div>`;
      }
      if (data.shap_summary_img) {
        vizHtml += `<div class="mb-3"><b>Model Explainability (SHAP):</b><br><img src="${data.shap_summary_img}" alt="SHAP Summary" class="img-fluid" style="max-width:600px;"></div>`;
      }
      if (vizHtml) {
        metricsHtml += `<div class="card p-3 mb-3"><h5 class="mb-3">Visualizations</h5>${vizHtml}</div>`;
      }
      statusDiv.innerHTML = metricsHtml;
    } catch (err) {
      statusDiv.innerText = 'Upload failed. Network error.';
    }
  };
}); 