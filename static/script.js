document.addEventListener('DOMContentLoaded', () => {
  const brandNameEl = document.getElementById('brand-name');
  const modelLabelEl = document.getElementById('model-label');
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const previewImage = document.getElementById('preview-image');
  const resultLabelEl = document.getElementById('result-label');
  const resultValueEl = document.getElementById('result-value');
  const resultDisplay = document.getElementById('result-display');
  const langToggleBtn = document.getElementById('langToggle');

  const translations = {
    en: {
      brand: "Digit Recognizer",
      model_label: "Model:",
      upload_prompt_drop: "Drag & drop an image here, or",
      upload_prompt_browse: "click to upload",
      prediction_label: "Prediction:",
      recognizing: "Recognizing...",
    },
    zh: {
      brand: "手写数字识别",
      model_label: "模型：",
      upload_prompt_drop: "将图片拖到此处，或",
      upload_prompt_browse: "点击上传",
      prediction_label: "识别结果：",
      recognizing: "识别中..."
    }
  };

  let currentLang = localStorage.getItem('lang') || (navigator.language.startsWith('zh') ? 'zh' : 'en');
  applyTranslations(currentLang);

  langToggleBtn.addEventListener('click', () => {
    currentLang = (currentLang === 'en') ? 'zh' : 'en';
    localStorage.setItem('lang', currentLang);
    applyTranslations(currentLang);
  });

  function applyTranslations(lang) {
    const t = translations[lang];
    brandNameEl.textContent = t.brand;
    modelLabelEl.textContent = t.model_label;

    const dropOrSpan = dropZone.querySelector('[data-i18n-key="drop_or"]');
    const browseSpan = dropZone.querySelector('[data-i18n-key="browse"]');
    if (dropOrSpan && browseSpan) {
      dropOrSpan.textContent = t.upload_prompt_drop;
      browseSpan.textContent = t.upload_prompt_browse;
    }

    if (resultLabelEl.textContent) {
      resultLabelEl.textContent = resultValueEl.textContent ? t.prediction_label : t.recognizing;
    }

    langToggleBtn.textContent = (lang === 'en') ? '中文' : 'EN';
  }

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('border-blue-500');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('border-blue-500');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('border-blue-500');
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
      e.dataTransfer.clearData();
    }
  });

  dropZone.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
    }
  });

  function handleFile(file) {
    showImagePreview(file);
    sendForRecognition(file);
  }

  function showImagePreview(file) {
    const reader = new FileReader();
    reader.onload = () => {
      previewImage.src = reader.result;
      previewImage.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }

  function sendForRecognition(file) {
    const t = translations[currentLang];
    resultLabelEl.textContent = t.recognizing;
    resultValueEl.textContent = "";

    const modelName = document.getElementById('model-select').value;
    const formData = new FormData();
    formData.append('image', file);
    formData.append('model', modelName);

    fetch('/predict', {
      method: 'POST',
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      resultLabelEl.textContent = t.prediction_label;
      resultValueEl.textContent = data.prediction || "Error";
    })
    .catch(err => {
      console.error('识别出错：', err);
      resultLabelEl.textContent = t.prediction_label;
      resultValueEl.textContent = "Error";
    });
  }
});
