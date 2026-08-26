// 종목(group_select) -> 생성일자(date_select) -> 숨은 file input, 그리고 기간(period) 입력
// 기본값 채우기까지 담당하는 공용 스크립트. 백테스트/daily/히트맵 등 로컬 CSV를 고르는
// 모든 페이지의 템플릿에 templates/_csv_file_selector.html을 통해 포함된다.
(function () {
  const data = window.CSV_SELECTOR_DATA || {};
  const groupMap = {};
  (data.groups || []).forEach(g => { groupMap[g.code] = g.items; });

  const groupSelect = document.getElementById('group_select');
  const dateSelect = document.getElementById('date_select');
  const fileInput = document.getElementById('file');
  const periodInput = document.getElementById('period');

  if (!groupSelect || !dateSelect || !fileInput) return;

  function findItem(code, filename) {
    return (groupMap[code] || []).find(item => item.filename === filename);
  }

  function applyMaxDaysAsPeriodDefault() {
    const item = findItem(groupSelect.value, dateSelect.value);
    if (item && periodInput) periodInput.value = item.max_days;
  }

  function populateDateSelect(key, preselectFilename, resetPeriod) {
    dateSelect.innerHTML = '';
    const items = groupMap[key] || [];
    if (items.length === 0) {
      dateSelect.disabled = true;
      dateSelect.innerHTML = '<option value="">데이터 없음</option>';
      return;
    }
    dateSelect.disabled = false;
    items.forEach(item => {
      const opt = document.createElement('option');
      opt.value = item.filename;
      opt.textContent = item.created_display + ' (최대 ' + item.max_days + '일)';
      if (item.filename === preselectFilename) opt.selected = true;
      dateSelect.appendChild(opt);
    });
    fileInput.value = dateSelect.value;
    if (resetPeriod) applyMaxDaysAsPeriodDefault();
  }

  // 사용자가 직접 종목/생성일자를 바꾸면 기간 입력을 그 파일의 최대범위로 리셋한다.
  groupSelect.addEventListener('change', () => populateDateSelect(groupSelect.value, null, true));
  dateSelect.addEventListener('change', () => {
    fileInput.value = dateSelect.value;
    applyMaxDaysAsPeriodDefault();
  });

  // 최초 로드(폼 재제출/링크 클릭 등)는 URL에 이미 기간이 있으면 건드리지 않고,
  // 없으면(첫 방문 등) 선택된 파일의 최대범위를 기간의 기본값으로 채워준다.
  const preselected = data.selectedFile || data.defaultFile;
  if (preselected) {
    const parts = preselected.match(/^(\d+)_/);
    if (parts) {
      groupSelect.value = parts[1];
      populateDateSelect(parts[1], preselected, !data.hasExplicitPeriod);
    }
  }

  const form = fileInput.closest('form');
  if (form) {
    form.addEventListener('submit', (e) => {
      if (!fileInput.value) {
        e.preventDefault();
        alert('종목과 생성일자를 모두 선택해주세요.');
      }
    });
  }
})();
