// Progressive enhancement: commands remain readable/selectable without JS.
document.querySelectorAll('main.linuxday pre').forEach((pre, index) => {
  const wrapper = document.createElement('div');
  wrapper.className = 'code-block';
  pre.before(wrapper);
  const toolbar = document.createElement('div');
  toolbar.className = 'code-toolbar';
  const button = document.createElement('button');
  button.className = 'copy-code';
  button.type = 'button';
  button.textContent = 'Copia';
  button.setAttribute('aria-label', `Copia il blocco di comandi ${index + 1}`);
  button.setAttribute('aria-live', 'polite');
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(pre.textContent);
      button.textContent = 'Copiato';
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(pre);
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = 'Selezionato: copia dal menu';
    }
    setTimeout(() => { button.textContent = 'Copia'; }, 2500);
  });
  toolbar.append(button);
  wrapper.append(toolbar, pre);
});
document.querySelectorAll('main.linuxday table').forEach(table => {
  const wrapper = document.createElement('div');
  wrapper.className = 'table-scroll';
  wrapper.tabIndex = 0;
  wrapper.setAttribute('role', 'region');
  wrapper.setAttribute('aria-label', 'Tabella scorrevole');
  table.before(wrapper);
  wrapper.append(table);
});
