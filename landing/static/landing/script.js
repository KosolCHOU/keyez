// Year
document.getElementById('y').textContent = new Date().getFullYear()

const themeStorageKey = 'keyez-theme'
const themeToggle = document.getElementById('themeToggle')
const prefersColorScheme = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null

function getStoredTheme(){
  try{
    return localStorage.getItem(themeStorageKey)
  }catch(_){
    return null
  }
}

function setStoredTheme(value){
  try{
    localStorage.setItem(themeStorageKey, value)
  }catch(_){
    // ignore write errors (e.g., Safari private mode)
  }
}

function applyTheme(theme){
  document.body.dataset.theme = theme
  document.documentElement.style.colorScheme = theme
  if(themeToggle){
    const label = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'
    themeToggle.textContent = theme === 'dark' ? '☀' : '🌙'
    themeToggle.setAttribute('aria-label', label)
  }
}

let activeTheme = document.body.dataset.theme || 'dark'
const storedTheme = getStoredTheme()

if(storedTheme){
  activeTheme = storedTheme
}else if(prefersColorScheme){
  activeTheme = prefersColorScheme.matches ? 'dark' : 'light'
}

applyTheme(activeTheme)

if(themeToggle){
  themeToggle.addEventListener('click', ()=>{
    activeTheme = activeTheme === 'dark' ? 'light' : 'dark'
    setStoredTheme(activeTheme)
    applyTheme(activeTheme)
  })
}

if(prefersColorScheme){
  const handleSchemeChange = (event)=>{
    if(getStoredTheme()) return
    activeTheme = event.matches ? 'dark' : 'light'
    applyTheme(activeTheme)
  }

  if(typeof prefersColorScheme.addEventListener === 'function'){
    prefersColorScheme.addEventListener('change', handleSchemeChange)
  }else if(typeof prefersColorScheme.addListener === 'function'){
    prefersColorScheme.addListener(handleSchemeChange)
  }
}

// Simple SingKhmer -> Khmer demo mapping (illustrative only)
const demoMap = new Map([
  ['sl', ['ស្រឡាញ់','ស្លាញ់','ស្រលាញ់','ស្រឡាញ','ស្រឡាញ់ណា']],
  ['kom', ['កុំ','កម','កុំ់','កុំ៕','កុំទៅ']],
  ['tv', ['ទៅ','តូវ','តូវ៉','តូវ៕','ទៅណា']],
  ['tov', ['ទៅ','តូវ','តូវ៉','តូវ៕','ទៅណា']],
  ['nak', ['អ្នក','ណាក់','នាក់','នាគ','ណ៎ក']],
  ['sok', ['សុខ','សុក','សុគ','សូក','ស៉ុក']],
  ['som', ['សុំ', 'សូម','សម','សុំ់','សុំ៕','សុំទៅ']],
  ['sabai', ['សប្បាយ','សបាយ','សាបាយ','សប្បាយ់','សបា នៅ']],
  ['chum', ['ជំរាប','ជុំ','ជឹម','ចុំ','ជំ']],
  ['reap', ['រាប','រៀប','រាបសារ','រៀបរាប','រាបរ']],
  ['suor', ['សួស្ដី','សួរ','សួស្តី','សួស្តី់','សួសរ']],
  ['arun', ['អរុញ','អរុន','អរុន៍','អរុន់','អរុ']],
  ['susdei', ['សួស្ដី','សួស្តី','សុស្ដី','សុស្តី','សួសឌី']],
  ['khnhom', ['ខ្ញុំ','ខញុំ','ខ្ញុ','ខ្ញុំ៕','ខ្ញុំឯង']],
  ['mean', ['មាន','មាន់','មាន់់','មន','មាន៕']],
])

// --- AJOUTS pour la phrase de démo KeyEZ ---
demoMap.set('saum', ['សូម']);
demoMap.set('chhnam', ['ឆ្នាំ']);
demoMap.set('som', ['សូម']);
demoMap.set('nenam', ['ណែនាំ']);
demoMap.set('vithi', ['វិធី']);
demoMap.set('.', ['។', '។', '។']);
demoMap.set('thmei', ['ថ្មី']);
demoMap.set('yg', ['យើង']);
demoMap.set('keyez', ['KeyEZ']);
demoMap.set('chea', ['ជា']);
demoMap.set('pheasaea', ['ភាសា']);
demoMap.set('khmer', ['ខ្មែរ']);
demoMap.set('bangkeut', ['បង្កើត']);
demoMap.set('vithi', ['វិធី']);
demoMap.set('daembi', ['ដើម្បី']);
demoMap.set('sarser', ['សរសេរ']);
demoMap.set('khmer', ['ខ្មែរ']);        // déjà présent en provider, on double pour robustesse
demoMap.set('rloun', ['រលូន']);
demoMap.set('chhlatvei', ['ឆ្លាតវៃ']);
demoMap.set('sl', ['ស្រឡាញ់']);
demoMap.set('kom', ['កុំ']);
demoMap.set('tov', ['ទៅ']);
demoMap.set('ksk', ['ខ្មែរស្រឡាញ់ខ្មែរ']);
demoMap.set('.', ['។']);
demoMap.set('phlit', ['ផលិត']);
demoMap.set('daoy', ['ដោយ']);
demoMap.set('astroai', ['AstroAI']);
// ===== Vocab courant pour démo temps réel =====
// Salutations
demoMap.set('suosdei', ['សួស្តី','សួស្ដី','ជំរាបសួរ']);
demoMap.set('jomreabsuor', ['ជំរាបសួរ','សួស្តី']);
demoMap.set('akun', ['អរគុណ','អរគុណច្រើន']);
demoMap.set('sumto', ['សុំទោស','អភ័យទោស']);

// Pronoms & personnes
demoMap.set('khnhom', ['ខ្ញុំ']);
demoMap.set('anak', ['អ្នក']);
demoMap.set('yeung', ['យើង']);
demoMap.set('koat', ['គាត់']);
demoMap.set('kvam', ['ខ្ញំ់វាង']); // option démo
demoMap.set('bang', ['បង']);
demoMap.set('oun', ['ប្អូន']);
demoMap.set('mitt', ['មិត្ត']);
demoMap.set('kru', ['គ្រូ']);

// Verbes de base
demoMap.set('tver', ['ធ្វើ']);
demoMap.set('tverka', ['ធ្វើការ']);
demoMap.set('rean', ['រៀន']);
demoMap.set('tov', ['ទៅ']);
demoMap.set('mok', ['មក']);
demoMap.set('jol', ['ចូល']);
demoMap.set('chenh', ['ចេញ']);
demoMap.set('chang', ['ចង់']);
demoMap.set('ban', ['បាន']);
demoMap.set('kompong', ['កំពុង']);
demoMap.set('ning', ['នឹង']);
demoMap.set('nyam', ['ញ៉ាំ']);
demoMap.set('phek', ['ផឹក']);
demoMap.set('som', ['សុំ']);
demoMap.set('del', ['ដែល']);
demoMap.set('klach', ['ខ្លាច']); // exemple émotion

// Lieux / objets
demoMap.set('phteah', ['ផ្ទះ']);
demoMap.set('sala', ['សាលា']);
demoMap.set('psar', ['ផ្សារ']);
demoMap.set('ti krong', ['ទីក្រុង']);
demoMap.set('phnom', ['ភ្នំពេញ']);
demoMap.set('thanakier', ['ធនាគារ']);
demoMap.set('braek', ['ប្រាក់']);
demoMap.set('aba', ['ABA']);
demoMap.set('wing', ['Wing']);

// Temps / calendrier
demoMap.set('thngai', ['ថ្ងៃ']);
demoMap.set('sapada', ['សប្ដាហ៍']);
demoMap.set('khae', ['ខែ']);
demoMap.set('chhnam', ['ឆ្នាំ']);
demoMap.set('deljel', ['ព្រឹក']);
demoMap.set('tiatrov', ['ល្ងាច']); // approximatif pour démo

// Adjectifs / fréquences
demoMap.set('laor', ['ល្អ']);
demoMap.set('khlaing', ['ខ្លាំង']);
demoMap.set('yerk', ['យឺត']);
demoMap.set('reus', ['រូបស']) // démo (peut être retiré)
demoMap.set('sabay', ['សប្បាយ','សបាយ']);
demoMap.set('krai krai', ['ញឹកញាប់']);
demoMap.set('yok yok', ['ញឹកញាប់']); // alias

// Connecteurs utiles
demoMap.set('hai', ['ហើយ']);
demoMap.set('te', ['ទេ']);
demoMap.set('nae', ['ណែ']); // interj. démo
demoMap.set('ot', ['អត់']);

// Ponctuation khmère
demoMap.set('khtam', ['។']);
demoMap.set('somkol', ['ៈ']);

// Single letter mappings for handwriting recognition (English → Khmer consonants/vowels)
demoMap.set('a', ['អា','អ','អ៊ា']);
demoMap.set('b', ['ប','ប៉','ប៊']);
demoMap.set('c', ['ច','ឆ','ក']);
demoMap.set('d', ['ដ','ឌ','ទ']);
demoMap.set('e', ['អេ','អ៊ី','ឯ']);
demoMap.set('f', ['ហ្វ','ផ','ហ']);
demoMap.set('g', ['គ','ក','ហ្គ']);
demoMap.set('h', ['ហ','ហ៊','ហ៏']);
demoMap.set('i', ['អ៊ី','អិ','ឥ']);
demoMap.set('j', ['ច','ជ','ជ៊']);
demoMap.set('k', ['ក','ខ','គ']);
demoMap.set('l', ['ល','អ','ឡ']);
demoMap.set('m', ['ម','ម៉','អិម']);
demoMap.set('n', ['ន','ណ','ន៉']);
demoMap.set('o', ['អូ','អោ','អ៊ូ']);
demoMap.set('p', ['ប','ព','ផ']);
demoMap.set('q', ['ក','ក្យូ','ខ']);
demoMap.set('r', ['រ','ឫ','រ៉']);
demoMap.set('s', ['ស','ស្រ','ស៊']);
demoMap.set('t', ['ត','ថ','ទ']);
demoMap.set('u', ['អ៊ូ','អុ','ឧ']);
demoMap.set('v', ['វ','វ៉','វី']);
demoMap.set('w', ['វ','វ៉','ដាប់ប៊ូយូ']);
demoMap.set('x', ['ក្ស','អិច','ឃ']);
demoMap.set('y', ['យ','យ៉','យ៊ី']);
demoMap.set('z', ['ហ្ស','ស','ហ្សេត']);


const input = document.getElementById('latinInput')
const out = document.getElementById('khOutput')
const cand = document.getElementById('candidates')

const animatedElements = Array.from(new Set([
  ...document.querySelectorAll('[data-animate]'),
  ...document.querySelectorAll('.card')
]))

animatedElements.forEach(el=>{
  if(!el.classList.contains('animate-on-scroll')){
    el.classList.add('animate-on-scroll')
  }
})

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')

if(prefersReducedMotion.matches || typeof IntersectionObserver === 'undefined'){
  animatedElements.forEach(el=>el.classList.add('is-visible'))
}else if(animatedElements.length){
  const observer = new IntersectionObserver((entries)=>{
    entries.forEach(entry=>{
      if(entry.isIntersecting){
        entry.target.classList.add('is-visible')
        observer.unobserve(entry.target)
      }
    })
  },{threshold:0.18, rootMargin:'0px 0px -40px'})

  animatedElements.forEach(el=>{
    const rect = el.getBoundingClientRect()
    if(rect.top <= window.innerHeight * 0.85){
      el.classList.add('is-visible')
    }else{
      observer.observe(el)
    }
  })
}

function tokenize(str){
  return str.trim().split(/\s+/)
}

function renderCandidates(word){
  if(!cand) return
  cand.innerHTML = ''
  const key = (word||'').toLowerCase()
  const options = demoMap.get(key) || []
  options.slice(0,5).forEach((opt, i)=>{
    const b = document.createElement('button')
    b.className = 'kbd-btn px-3 py-2 rounded-xl hover:bg-white/10'
    b.textContent = `${i+1}. ${opt}`
    b.addEventListener('click', ()=>acceptCandidate(opt))
    cand.appendChild(b)
  })
}

function acceptCandidate(text){
  if(!out || !input || !cand) return
  const current = out.value || ''
  out.value = (current.trim() + ' ' + text).trim()
  cand.innerHTML = ''
  input.value = ''
  input.focus()
}

if(input){
  input.addEventListener('input', e=>{
    const words = tokenize(e.target.value)
    if(words.length){ renderCandidates(words[words.length-1]) }
  })
}

// number keys 1-3 to pick candidate (context-aware: main demo, social demo, or canvas)
window.addEventListener('keydown', (e)=>{
  if(!/^[1-3]$/.test(e.key)) return
  const idx = Number(e.key)-1
  const ae = document.activeElement
  const fbC = document.getElementById('fbCandidates')
  const canvasC = document.getElementById('canvasCandidates')
  let target = null
  
  // Check which candidate bar should respond
  if(ae && ae.id === 'fbInput' && fbC){
    target = fbC
  } else if(ae && ae.id === 'canvasOutput' && canvasC){
    target = canvasC
  } else if(cand){
    target = cand
  }
  
  if(!target) return
  const children = Array.from(target.children)
  if(children[idx]) children[idx].click()
})

// Canvas handwriting demo with auto-recognition
const pad = document.getElementById('pad')
const canvasOutput = document.getElementById('canvasOutput')
const canvasCandidates = document.getElementById('canvasCandidates')
const currentWordDiv = document.getElementById('currentWord')
const liveRecognitionDiv = document.getElementById('liveRecognition')
const recognizedCharSpan = document.getElementById('recognizedChar')
const recognizedConfidenceSpan = document.getElementById('recognizedConfidence')
const ctx = pad ? pad.getContext('2d') : null

if(ctx){
  ctx.lineWidth = 4
  ctx.lineCap = 'round'
  ctx.strokeStyle = '#000'
}

let drawing = false
let lx = 0
let ly = 0
let strokeCount = 0
let recognitionTimeout = null
let currentWord = ''  // Building up the word
let hasStrokes = false

function pos(e){ 
  const r = pad.getBoundingClientRect()
  const t = e.touches ? e.touches[0] : e
  return {x: t.clientX - r.left, y: t.clientY - r.top}
}

function start(e){ 
  drawing = true
  hasStrokes = true
  const p = pos(e)
  lx = p.x
  ly = p.y
  strokeCount++
  
  // Cancel any pending recognition
  if(recognitionTimeout){
    clearTimeout(recognitionTimeout)
  }
}

function move(e){ 
  if(!drawing) return
  const p = pos(e)
  ctx.beginPath()
  ctx.moveTo(lx, ly)
  ctx.lineTo(p.x, p.y)
  ctx.stroke()
  lx = p.x
  ly = p.y
}

function end(){ 
  if(!drawing) return
  drawing = false
  
  // Auto-recognize after user stops drawing (500ms delay)
  if(recognitionTimeout){
    clearTimeout(recognitionTimeout)
  }
  
  if(hasStrokes){
    recognitionTimeout = setTimeout(()=>{
      autoRecognizeLetter()
    }, 500)  // Wait 500ms after last stroke
  }
}

if(pad){
  pad.addEventListener('mousedown', start)
  pad.addEventListener('mousemove', move)
  window.addEventListener('mouseup', end)
  pad.addEventListener('touchstart', (e)=>{start(e); e.preventDefault()}, {passive:false})
  pad.addEventListener('touchmove', (e)=>{move(e); e.preventDefault()}, {passive:false})
  window.addEventListener('touchend', end)
}

// Clear canvas (just the drawing, not the word)
const clearBtn = document.getElementById('clearPad')
if(clearBtn){
  clearBtn.addEventListener('click', ()=>{
    clearCanvas()
  })
}

// Clear the entire word being built
const clearWordBtn = document.getElementById('clearWord')
if(clearWordBtn){
  clearWordBtn.addEventListener('click', ()=>{
    currentWord = ''
    updateCurrentWord()
    clearCanvas()
    if(canvasCandidates){
      canvasCandidates.innerHTML = ''
    }
  })
}

// Add space to move to next word
const addSpaceBtn = document.getElementById('addSpace')
if(addSpaceBtn){
  addSpaceBtn.addEventListener('click', ()=>{
    if(currentWord){
      // Add current word to output
      const current = canvasOutput.value || ''
      canvasOutput.value = (current.trim() + ' ' + currentWord).trim() + ' '
      
      // Reset for next word
      currentWord = ''
      updateCurrentWord()
      clearCanvas()
      if(canvasCandidates){
        canvasCandidates.innerHTML = ''
      }
    }
  })
}

function clearCanvas(){
  if(ctx){
    ctx.clearRect(0, 0, pad.width, pad.height)
  }
  hasStrokes = false
  strokeCount = 0
  hideLiveRecognition()
}

function updateCurrentWord(){
  if(currentWordDiv){
    if(currentWord){
      currentWordDiv.innerHTML = `<span class="font-bold">${currentWord}</span>`
    } else {
      currentWordDiv.innerHTML = '<span class="opacity-50">Type letters to build words...</span>'
    }
  }
}

function showLiveRecognition(char, confidence){
  if(liveRecognitionDiv && recognizedCharSpan && recognizedConfidenceSpan){
    recognizedCharSpan.textContent = char
    recognizedConfidenceSpan.textContent = `${(confidence * 100).toFixed(0)}%`
    liveRecognitionDiv.style.opacity = '1'
  }
}

function hideLiveRecognition(){
  if(liveRecognitionDiv){
    liveRecognitionDiv.style.opacity = '0'
  }
}

// Auto recognition function - called after drawing stops
async function autoRecognizeLetter(){
  if(!pad || !ctx || !hasStrokes) return
  
  try {
    // Convert canvas to base64 PNG
    const imageData = pad.toDataURL('image/png')
    
    console.log('Sending canvas data, size:', imageData.length, 'chars')
    console.log('Canvas dimensions:', pad.width, 'x', pad.height)
    console.log('Canvas has strokes:', hasStrokes, 'stroke count:', strokeCount)
    
    // Send to backend with debug flag
    const response = await fetch('/predict/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        image: imageData,
        debug: true  // Enable debug mode to see preprocessed image
      })
    })
    
    const result = await response.json()
    
    console.log('Prediction result:', result)
    
    // If debug image is available, log it
    if(result.preprocessed_image){
      console.log('Preprocessed image (what model sees):', result.preprocessed_image)
      console.log('You can paste this in browser address bar to view it')
    }
    
    if(result.success && result.predictions && result.predictions.length > 0){
      // Get the best English letter prediction
      const bestPrediction = result.predictions[0]
      const recognizedLetter = bestPrediction.char.toLowerCase()
      const confidence = bestPrediction.confidence
      
      // Show live recognition feedback
      showLiveRecognition(recognizedLetter, confidence)
      
      // Add letter to current word
      currentWord += recognizedLetter
      updateCurrentWord()
      
      // Clear canvas for next letter
      setTimeout(()=>{
        clearCanvas()
      }, 800)  // Show recognition for 800ms before clearing
      
      // Get Khmer suggestions for the current word
      updateKhmerSuggestions()
      
    } else {
      console.error('Prediction failed:', result.error)
    }
    
  } catch(error) {
    console.error('Recognition error:', error)
  }
}

// Update Khmer suggestions based on current word
function updateKhmerSuggestions(){
  if(!currentWord || !canvasCandidates) return
  
  // Look up current word in demoMap
  const key = currentWord.toLowerCase()
  const khmerOptions = demoMap.get(key) || []
  
  if(khmerOptions.length > 0){
    // Create candidates from Khmer options
    const candidates = khmerOptions.slice(0, 3).map(khmer => ({
      char: khmer,
      confidence: 0.9  // Mock confidence for display
    }))
    renderCanvasCandidates(candidates)
  } else {
    // No direct match - show letter-by-letter Khmer
    const letters = currentWord.split('')
    const khmerLetters = letters.map(letter => {
      const opts = demoMap.get(letter) || []
      return opts[0] || letter
    }).join('')
    
    if(khmerLetters){
      renderCanvasCandidates([
        { char: khmerLetters, confidence: 0.7 },
        { char: currentWord, confidence: 0.5 },  // Fallback to English
        { char: '', confidence: 0 }
      ])
    } else {
      canvasCandidates.innerHTML = ''
    }
  }
}

function renderCanvasCandidates(results){
  if(!canvasCandidates) return
  canvasCandidates.innerHTML = ''
  
  // Always show exactly 3 slots (iOS style)
  const slots = [
    results[1]?.char || '',  // Left
    results[0]?.char || '',  // Middle (best)
    results[2]?.char || ''   // Right
  ]
  
  for(let i = 0; i < 3; i++){
    const txt = slots[i]
    const b = document.createElement('button')
    b.type = 'button'
    b.className = 'slot' + (i === 1 && txt ? ' primary' : '')
    b.setAttribute('data-idx', String(i))
    b.setAttribute('role', 'option')
    b.setAttribute('aria-selected', i === 1 && txt ? 'true' : 'false')
    
    if(txt){
      b.textContent = txt
      b.addEventListener('click', ()=>acceptCanvasCandidate(txt))
    } else {
      b.textContent = ''
      b.setAttribute('disabled', '')
      b.setAttribute('aria-hidden', 'true')
    }
    canvasCandidates.appendChild(b)
  }
}

function acceptCanvasCandidate(text){
  if(!canvasOutput || !text) return
  
  // Add selected Khmer word to output
  const current = canvasOutput.value || ''
  canvasOutput.value = (current.trim() + ' ' + text).trim() + ' '
  
  // Reset for next word
  currentWord = ''
  updateCurrentWord()
  clearCanvas()
  
  // Clear candidates after selection
  if(canvasCandidates){
    canvasCandidates.innerHTML = ''
  }
  
  canvasOutput.focus()
}

// Keyboard navigation for canvas candidates
if(canvasOutput){
  canvasOutput.addEventListener('keydown', (e)=>{
    if(!canvasCandidates) return
    
    // Number keys 1-3 select candidates
    if(/^[1-3]$/.test(e.key)){
      const idx = Number(e.key) - 1
      const btn = canvasCandidates.querySelector(`.slot[data-idx="${idx}"]:not([disabled])`)
      if(btn){ 
        btn.click()
        e.preventDefault()
      }
      return
    }
    
    // Enter accepts middle (best) candidate (changed from Space to allow normal space typing for English)
    if(e.key === 'Enter'){
      const mid = canvasCandidates.querySelector('.slot[data-idx="1"]:not([disabled])')
      if(mid){ 
        mid.click()
        e.preventDefault()
      }
      return
    }
    
    // Escape clears candidates
    if(e.key === 'Escape'){
      canvasCandidates.innerHTML = ''
    }
  })
}

// Language toggle
const langStorageKey = 'keyez-lang'
const langBtn = document.getElementById('langToggle')

function getStoredLang(){
  try{
    return localStorage.getItem(langStorageKey) || 'en'
  }catch(_){
    return 'en'
  }
}

function setStoredLang(value){
  try{ localStorage.setItem(langStorageKey, value) }catch(_){}
}

function applyLang(lang){
  document.body.dataset.lang = lang
  document.documentElement.lang = lang === 'en' ? 'en' : 'km'
  if(langBtn){ langBtn.textContent = lang === 'en' ? 'EN' : 'ខ្មែរ' }
  const elements = document.querySelectorAll('[data-lang-en][data-lang-km]')
  elements.forEach(el=>{
    const text = lang === 'en' ? el.dataset.langEn : el.dataset.langKm
    if(text){
      if(el.tagName === 'H1' && text.includes('.')){
        const parts = text.split('.').filter(p=>p.trim())
        el.innerHTML = parts.map(p=>p.trim()+'.').join('<br/>')
      }else{
        el.textContent = text
      }
    }
  })
  if(lang === 'km'){
    document.body.classList.add('font-[Noto_Sans_Khmer]')
    document.body.classList.remove('font-[Inter]')
  }else{
    document.body.classList.add('font-[Inter]')
    document.body.classList.remove('font-[Noto_Sans_Khmer]')
  }
}

let activeLang = getStoredLang()
applyLang(activeLang)
if(langBtn){
  langBtn.addEventListener('click', ()=>{
    activeLang = activeLang === 'en' ? 'km' : 'en'
    setStoredLang(activeLang)
    applyLang(activeLang)
  })
}

// Active navigation tracking
const navLinks = document.querySelectorAll('header nav a[href^="#"]')
const sections = Array.from(navLinks).map(link => {
  const href = link.getAttribute('href')
  return { link, section: document.querySelector(href), id: href }
})

function updateActiveNav() {
  const scrollPos = window.scrollY + 150 // offset for fixed header
  
  let currentSection = null
  
  // Find the current section based on scroll position
  for (let i = sections.length - 1; i >= 0; i--) {
    const { section, id } = sections[i]
    if (section && section.offsetTop <= scrollPos) {
      currentSection = id
      break
    }
  }
  
  // Update active class on nav links
  navLinks.forEach(link => {
    const href = link.getAttribute('href')
    if (href === currentSection) {
      link.classList.add('active')
    } else {
      link.classList.remove('active')
    }
  })
}

// Update on scroll with throttle
let ticking = false
window.addEventListener('scroll', () => {
  if (!ticking) {
    window.requestAnimationFrame(() => {
      updateActiveNav()
      ticking = false
    })
    ticking = true
  }
})

// Update on load
window.addEventListener('load', updateActiveNav)
// Update immediately
updateActiveNav()
// Social Demo (Facebook caption) — keyboard + candidate bar
function initSocialDemo(){
  const fbInput = document.getElementById('fbInput')
  const fbCandidates = document.getElementById('fbCandidates')
  const fbKeyboard = document.getElementById('fbKeyboard')
  if(!fbInput || !fbCandidates || !fbKeyboard) return

  // Build virtual keyboard
  const row1 = 'qwertyuiop'.split('')
  const row2 = 'asdfghjkl'.split('')
  // Move space to its own bottom row to avoid wrap and fill width like iOS
  const row3 = ['shift','z','x','c','v','b','n','m','⌫']
  const row4 = ['123','🌐','🎤','space','return']
  let shifted = false
  
  // Map buttons for physical keyboard highlight (declare early)
  const keyButtons = new Map()

  function keyEl(label, classes='', span=1){
    const b = document.createElement('button')
    b.type = 'button'
    // Base keyboard key styling
    let baseClasses = 'kbd-key'
    
    // Add special classes for different key types
    if(label === 'shift' || label === '⌫') baseClasses += ' kbd-key-wide'
    else if(label === 'space') baseClasses += ' kbd-key-space'
    else if(label === 'return') baseClasses += ' kbd-key-enter'
    
    // Add Tailwind grid span classes if needed
    if(span > 1) baseClasses += ` col-span-${span}`
    
    b.className = baseClasses + (classes ? ' ' + classes : '')
    // Always set data-key to lowercase for alpha keys
    if(/^[a-z]$/i.test(label)){
      b.setAttribute('data-key', label.toLowerCase())
      b.textContent = shifted ? label.toUpperCase() : label.toLowerCase()
    }else{
      b.setAttribute('data-key', label)
      b.textContent = label
    }
    return b
  }

  function insertAtCursor(str){
    const start = fbInput.selectionStart
    const end = fbInput.selectionEnd
    const val = fbInput.value
    fbInput.value = val.slice(0,start) + str + val.slice(end)
    const pos = start + str.length
    fbInput.setSelectionRange(pos,pos)
    fbInput.dispatchEvent(new Event('input',{bubbles:true}))
    fbInput.focus()
  }

  function handleKey(label){
    if(label === 'shift'){
      shifted = !shifted
      updateKeyCase()
      return
    }
    if(label === '⌫'){
      // backspace
      const start = fbInput.selectionStart
      const end = fbInput.selectionEnd
      if(start === end && start>0){
        fbInput.setSelectionRange(start-1,end)
      }
      insertAtCursor('')
      return
    }
    if(label === 'space'){
      insertAtCursor(' ')
      return
    }
    if(label === 'return'){
      insertAtCursor('\n')
      return
    }
    if(label === '123' || label === '🌐' || label === '🎤'){
      // placeholders for layout realism
      return
    }
    // For regular letter keys, insert with current shift state
    if(/^[a-z]$/i.test(label)){
      insertAtCursor(shifted? label.toUpperCase(): label)
      // Auto-reset shift after typing one character (like mobile keyboards)
      if(shifted){
        shifted = false
        updateKeyCase()
      }
    }else{
      insertAtCursor(label)
    }
  }

  const spanMap = new Map([
    ['shift',2],
    ['⌫',1],
    ['space',4],
    ['return',2],
    ['123',2],
    ['🌐',1],
    ['🎤',1],
  ])

  function addRow(keys, config=[]) {
    const rowDiv = document.createElement('div')
    rowDiv.className = 'flex gap-1 justify-center'
    
    keys.forEach((k, i)=>{
      const btn = keyEl(k, '')
      btn.setAttribute('data-key', k)
      btn.addEventListener('click', ()=>handleKey(k))
      rowDiv.appendChild(btn)
    })
    
    fbKeyboard.appendChild(rowDiv)
  }

  // Build keyboard layout with rows
  fbKeyboard.classList.add('space-y-1')
  addRow(row1)
  // Indent row2 slightly for realism
  const row2Div = document.createElement('div')
  row2Div.className = 'flex gap-1 justify-center px-4'
  row2.forEach(k=>{
    const btn = keyEl(k, '')
    btn.setAttribute('data-key', k)
    btn.addEventListener('click', ()=>handleKey(k))
    row2Div.appendChild(btn)
  })
  fbKeyboard.appendChild(row2Div)
  addRow(row3)
  addRow(row4)

  // Populate the keyButtons map with all created buttons
  fbKeyboard.querySelectorAll('button[data-key]').forEach(b=>{
    keyButtons.set(b.getAttribute('data-key'), b)
  })

  function setPressed(label, isDown){
    const b = keyButtons.get(label)
    if(!b) return
    if(isDown){
      b.classList.add('pressed')
      // Fallback removal in case keyup is missed
      if(b._pressedTimer) clearTimeout(b._pressedTimer)
      b._pressedTimer = setTimeout(()=>b.classList.remove('pressed'), 250)
    }else{
      b.classList.remove('pressed')
      if(b._pressedTimer){ clearTimeout(b._pressedTimer); b._pressedTimer=null }
    }
  }

  function updateKeyCase(){
    fbKeyboard.querySelectorAll('button[data-key]').forEach(b=>{
      const v = b.getAttribute('data-key')
      if(v && /^[a-z]$/i.test(v)){
        b.textContent = shifted ? v.toUpperCase() : v.toLowerCase()
      }
    })
    // Update shift button visual state
    const shiftBtn = keyButtons.get('shift')
    if(shiftBtn){
      if(shifted){
        shiftBtn.classList.add('active-shift')
      }else{
        shiftBtn.classList.remove('active-shift')
      }
    }
  }

  // Reflect physical keyboard press when typing in the caption
  window.addEventListener('keydown', e=>{
    if(document.activeElement !== fbInput) return;
    
    // Sync physical Shift key with virtual keyboard
    if(e.key === 'Shift' && !shifted){
      shifted = true;
      updateKeyCase();
      setPressed('shift', true);
    }
    
    // Visual feedback for letter keys
    if(/^[a-z]$/i.test(e.key)){
      setPressed(e.key.toLowerCase(), true);
    }
    
    // Visual feedback for special keys
    if(e.key === 'Backspace'){
      setPressed('⌫', true);
    }
    if(e.key === ' ' || e.code === 'Space'){
      setPressed('space', true);
    }
    if(e.key === 'Enter'){
      setPressed('return', true);
    }
    
    // 1/2/3 choisissent directement
    if(/^[1-3]$/.test(e.key)){
      const idx = Number(e.key)-1;
      const btn = fbCandidates.querySelector(`.slot[data-idx="${idx}"]:not([disabled])`);
      if(btn){ btn.click(); e.preventDefault(); }
      return;
    }
    // Enter = accept middle (best) candidate (was Space; Space now inserts spaces normally)
    if(e.key === 'Enter'){
      const mid = fbCandidates.querySelector('.slot[data-idx="1"]:not([disabled])');
      if(mid){ mid.click(); e.preventDefault(); return; }
    }
    // ←/→ changent la sélection visuelle, ou naviguent le curseur si pas de candidats
    if(e.key === 'ArrowRight' || e.key === 'ArrowLeft'){
      const slots = [...fbCandidates.querySelectorAll('.slot')].filter(el=>!el.disabled);
      // Only navigate candidates if there are any visible, otherwise allow normal cursor movement
      if(slots.length > 0){
        const max = slots.length-1;
        let i = candidateState.selectedIndex || 0;
        i = e.key==='ArrowRight' ? Math.min(i+1,max) : Math.max(i-1,0);
        candidateState.selectedIndex = i;
        fbCandidates.querySelectorAll('.slot').forEach((el,idx)=>{
          el.setAttribute('aria-selected', idx===i ? 'true' : 'false');
        });
        e.preventDefault();
      }
      // If no candidates, allow default cursor navigation (don't preventDefault)
    }
    // Esc = vider les candidats
    if(e.key === 'Escape'){ fbCandidates.innerHTML=''; }
  });

  window.addEventListener('keyup', (e)=>{
    if(document.activeElement !== fbInput) return
    
    // Sync physical Shift key release with virtual keyboard
    if(e.key === 'Shift'){
      shifted = false;
      updateKeyCase();
      setPressed('shift', false);
      return;
    }
    
    let label = null
    if(/^[a-z]$/i.test(e.key)) label = e.key.toLowerCase()
    else if(e.key === 'Backspace') label = '⌫'
    else if(e.key === ' ' || e.code === 'Space') label = 'space'
    else if(e.key === 'Enter') label = 'return'
    if(label){ setPressed(label, false) }
  })

  // Candidate rendering for social demo - USING AI MODEL
  async function renderFBCandidates(word){
    fbCandidates.innerHTML = '';
    const key = (word||'').toLowerCase();
    
    // Show loading state
    fbCandidates.innerHTML = '<div class="text-center text-slate-400 text-sm py-2">Translating...</div>';
    
    try {
      // Call transliteration API
      console.log('🔄 Calling API for:', key);
      const response = await fetch('/transliterate/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: key })
      });
      
      if(!response.ok){
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      console.log('✅ API response:', result);
      
      if(result.success && result.candidates){
        // API returns: [left (2nd best), middle (best), right (3rd best)]
        const slots = result.candidates;
        
        // Clear loading state
        fbCandidates.innerHTML = '';
        
        // Always show exactly 3 slots (iOS style fixed grid)
        for(let i=0;i<3;i++){
          const txt = slots[i] || '';
          const b = document.createElement('button');
          b.type = 'button';
          b.className = 'slot' + (i===1 && txt ? ' primary' : '');
          b.setAttribute('data-idx', String(i));
          b.setAttribute('role','option');
          b.setAttribute('aria-selected', i===1 && txt ? 'true' : 'false');
          if(txt){
            b.textContent = txt;
            b.addEventListener('click', ()=>acceptFBCandidate(txt));
          }else{
            b.textContent = '';
            b.setAttribute('disabled','');
            b.setAttribute('aria-hidden','true');
          }
          fbCandidates.appendChild(b);
        }
        candidateState = { items: slots, selectedIndex: slots.some(Boolean) ? 1 : 0 }; // best in middle
      } else {
        // Show error message instead of fallback
        console.error('❌ API returned unsuccessful response:', result);
        fbCandidates.innerHTML = '<div class="text-center text-red-400 text-sm py-2">Translation failed</div>';
      }
    } catch(error) {
      console.error('❌ Transliteration error:', error);
      // Show error message instead of fallback
      fbCandidates.innerHTML = '<div class="text-center text-red-400 text-sm py-2">Connection error</div>';
    }
  }

  const KHMER_INSERT_TRAILING_SPACE = false;

  function acceptFBCandidate(text){
    const val = fbInput.value;
    const parts = val.replace(/\n/g,' ').split(/\s+/);
    if(parts.length===0 || !parts[parts.length-1]){
      fbInput.value = (val + text).trim();
    }else{
      // remplace uniquement le dernier token latin par le khmer choisi
      parts[parts.length-1] = text;
      fbInput.value = parts.join(' ');
    }
    if(KHMER_INSERT_TRAILING_SPACE){
      fbInput.value += ' ';
    }
    fbCandidates.innerHTML = '';
    fbInput.focus();
  }

  fbInput.addEventListener('input', (e)=>{
    const val = e.target.value || '';
    const cursorPos = fbInput.selectionStart;
    
    // Immediate conversion for period - check if character before cursor is "."
    if(cursorPos > 0 && val[cursorPos - 1] === '.'){
      // Replace the period with Khmer period at cursor position
      const before = val.slice(0, cursorPos - 1);
      const after = val.slice(cursorPos);
      fbInput.value = before + '។' + after;
      // Restore cursor position after the Khmer period
      fbInput.setSelectionRange(cursorPos, cursorPos);
      fbCandidates.innerHTML = '';
      return;
    }
    
    const words = val.trim().split(/\s+/)
    const last = words[words.length-1] || ''
    
    if(last){ renderFBCandidates(last) } else { fbCandidates.innerHTML = '' }
  })

  // Add keyboard activation effect on focus
  fbInput.addEventListener('focus', ()=>{
    fbKeyboard.classList.add('keyboard-active')
    fbCandidates.classList.add('keyboard-active')
  })

  fbInput.addEventListener('blur', ()=>{
    fbKeyboard.classList.remove('keyboard-active')
    fbCandidates.classList.remove('keyboard-active')
  })

  // Flag: une barre sociale est active -> bloquer l'init générique
window.__FB_SOCIAL_DEMO_ACTIVE = true;
}

window.addEventListener('load', () => {
  // Initialize social demo first
  initSocialDemo();
  
  // Then auto-focus & scroll to social demo section
  const section = document.getElementById('social-demo');
  const fbInput = document.getElementById('fbInput');
  if (section) {
    setTimeout(() => {
      section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }
  if (fbInput) {
    setTimeout(() => {
      fbInput.focus();
      // Déclenche un rendu initial (si tu veux voir la barre vide en iOS mode)
      fbInput.dispatchEvent(new Event('input', { bubbles: true }));
    }, 300);
  }
});


// --- Provider interchangeable (mock aujourd’hui, API demain) ---
const CandidateProvider = {
  // Using AI Model API for transliteration
  async suggestRomanToKhmer(query) {
    const q = query.toLowerCase().trim();
    if (!q) return [];
    
    try {
      // Call the AI model API
      console.log('🔄 CandidateProvider calling API for:', q);
      const response = await fetch('/transliterate/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: q })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const result = await response.json();
      console.log('✅ CandidateProvider API response:', result);
      
      if (result.success && result.candidates) {
        // API returns [2nd, BEST, 3rd] - we want [BEST, 2nd, 3rd] for display
        return [result.candidates[1], result.candidates[0], result.candidates[2]].filter(Boolean);
      }
      
      // Fallback to common words on API failure
      return FALLBACKS.slice(0, 3);
    } catch (error) {
      console.error('❌ CandidateProvider error:', error);
      // Fallback to common words on error
      return FALLBACKS.slice(0, 3);
    }
  },
  
  // DEPRECATED - Old hardcoded dictionary (kept for reference only)
  _oldDict_DEPRECATED: {
      "saum": "សូម",
      "hi": "សួស្តី",
      "dg": "ដឹង",
      ".": "។",
      "som": "សូម",
      "chhnam": "ឆ្នាំ",
      "nenam": "ណែនាំ",
      "vithi": "វិធី",
      "thmei": "ថ្មី",
      "daembi": "ដើម្បី",
      "sarser": "សរសេរ",
      "khmer": "ខ្មែរ",
      "rloun": "រលូន",
      "chhlatvei": "ឆ្លាតវៃ",
      "phlit": "ផលិត",
      "daoy": "ដោយ",
      "astroai": "AstroAI",
      "sok": "សុខ",
      "sok sabay": "សុខសប្បាយ",
      "phnom": "ភ្នំពេញ",
      "penh": "ភ្នំពេញ",
      "srey": "ស្រី",
      "sala": "សាលា",
      "sruk": "ស្រុក",
      "yg": "យើង",
      "sl": "ស្រលាញ់",
      "ksk": "ខ្មែរស្រឡាញ់ខ្មែរ",
      "chea" : "ជា",
      ".": "។",
      "kom": "កុំ",
      "tov": "ទៅ",
      "nak": "អ្នក",
      "rloun": "រលូន",
      "srolanh": "ស្រឡាញ់",
      "suor": "សួស្ដី",
      "somtos": "សុំទោស",
      "chum reap suor": "ជំរាបសួរ",
      "som": "សូម",
      "pheasaea": "ភាសា",
      "arun": "អរគុណ",
      "susdei": "សួស្ដី",
      "bangkeut" : "បង្កើត",
      "daoy" : "ដោយ",
      "khnhom": "ខ្ញុំ",
      "mean": "មាន",
      "khmer": "ខ្មែរ",
      "sab": "សប្បាយ",
      "chomreab": "ជម្រើស",
      "suosdei":"សួស្តី","jomreabsuor":"ជំរាបសួរ","akun":"អរគុណ","sumto":"សុំទោស",
"khnhom":"ខ្ញុំ","anak":"អ្នក","yeung":"យើង","koat":"គាត់","bang":"បង","oun":"ប្អូន","mitt":"មិត្ត","kru":"គ្រូ",
"tver":"ធ្វើ","tverka":"ធ្វើការ","rean":"រៀន","tov":"ទៅ","mok":"មក","jol":"ចូល","chenh":"ចេញ",
"chang":"ចង់","ban":"បាន","kompong":"កំពុង","ning":"នឹង","nyam":"ញ៉ាំ","phek":"ផឹក","som":"សុំ","del":"ដែល","klach":"ខ្លាច",
"phteah":"ផ្ទះ","sala":"សាលា","psar":"ផ្សារ","ti krong":"ទីក្រុង","phnom":"ភ្នំពេញ","thanakier":"ធនាគារ","braek":"ប្រាក់","aba":"ABA","wing":"Wing",
"thngai":"ថ្ងៃ","sapada":"សប្ដាហ៍","khae":"ខែ","chhnam":"ឆ្នាំ","deljel":"ព្រឹក","tiatrov":"ល្ងាច",
"laor":"ល្អ","khlaing":"ខ្លាំង","yerk":"យឺត","sabay":"សប្បាយ","krai krai":"ញឹកញាប់","yok yok":"ញឹកញាប់",
"hai":"ហើយ","te":"ទេ","nae":"ណែ","ot":"អត់","khtam":"។","somkol":"ៈ", "thounho": "ធុញ"
    }
};

// ---- Toujours 3 suggestions : liste de secours très courante ----
const FALLBACKS = [
  'ខ្ញុំ','អ្នក','យើង','ទៅ','មក','បាន','កំពុង','នឹង',
  'សួស្តី','អរគុណ','សុំទោស','បាទ','ចាស',
  'បង','ប្អូន','មិត្ត','គ្រូ',
  'ផ្ទះ','សាលា','ធ្វើការ','រៀន','ញ៉ាំ','ផឹក',
  'ទីក្រុង','ភ្នំពេញ','ផ្សារ','ធនាគារ','ប្រាក់'
];

// --- Gestion UI ---
const fbCandidates = document.getElementById('fbCandidates');
let candidateState = { items: [], selectedIndex: 0 };

// util: debounce pour limiter les requêtes
function debounce(fn, delay=80){
  let t=null; return (...args)=>{ clearTimeout(t); t=setTimeout(()=>fn(...args),delay); };
}

// extrait le dernier token roman (singkhmer) depuis ta zone de saisie
function getLastRomanToken(text){
  // autorise lettres + apostrophes (pinyin-like)
  const m = text.match(/([a-zA-Z']+)$/);
  return m ? m[1] : "";
}

// remplace le dernier token par le khmer choisi + espace
function commitCandidateToInput(khmerWord){
  const input = document.getElementById('fbInput') || document.querySelector('textarea, input[type="text"]');
  if(!input) return;
  const before = input.value;
  const m = before.match(/([a-zA-Z']+)$/);
  if(m){
    input.value = before.slice(0, m.index) + khmerWord;
  }else{
    input.value = before + khmerWord;
  }
  input.focus();
  // reset barre
  renderCandidates([]);
}

// rend les pills
function renderCandidates(list){
  // Normalize to exactly 3 slots, and place best (index 0) in the middle visual slot
  const top3 = list.slice(0,3);
  const slots = [top3[1] || '', top3[0] || '', top3[2] || ''];
  candidateState.items = slots;
  candidateState.selectedIndex = slots.some(Boolean) ? 1 : 0;
  fbCandidates.innerHTML = "";

  for(let i=0;i<3;i++){
    const txt = slots[i] || '';
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'slot' + (i===1 && txt ? ' primary' : '');
    b.setAttribute('data-idx', String(i));
    b.setAttribute('role','option');
    b.setAttribute('aria-selected', i===1 && txt ? 'true' : 'false');
    if(txt){
      b.textContent = txt;
      b.addEventListener('click', ()=> commitCandidateToInput(txt));
    }else{
      b.textContent = '';
      b.setAttribute('disabled','');
      b.setAttribute('aria-hidden','true');
    }
    fbCandidates.appendChild(b);
  }
}

// met à jour la sélection visuelle
function updateSelectedCandidate(){
  [...fbCandidates.children].forEach((el, i)=>{
    el.setAttribute('aria-selected', i===candidateState.selectedIndex ? 'true' : 'false');
  });
  // auto-scroll le pill sélectionné dans la vue
  const selected = fbCandidates.children[candidateState.selectedIndex];
  if(selected){
    const rect = selected.getBoundingClientRect();
    const parentRect = fbCandidates.getBoundingClientRect();
    if(rect.right > parentRect.right) fbCandidates.scrollLeft += (rect.right - parentRect.right) + 12;
    if(rect.left < parentRect.left) fbCandidates.scrollLeft -= (parentRect.left - rect.left) + 12;
  }
}

// écoute la saisie pour déclencher les suggestions

// Universal input handler for period replacement and candidate bar
const onInputChanged = debounce(async (e)=>{
  // Use event target if available, else fallback
  const input = e && e.target ? e.target : (document.getElementById('fbInput') || document.querySelector('textarea, input[type="text"]'));
  if(!input) return;

  const cursorPos = input.selectionStart;
  const val = input.value;

  // ✅ 1. Si l'utilisateur tape ".", remplace automatiquement par "។"
  if (cursorPos > 0 && val[cursorPos - 1] === '.') {
    // Replace the period with Khmer period at cursor position
    const before = val.slice(0, cursorPos - 1);
    const after = val.slice(cursorPos);
    input.value = before + '។' + after;
    input.setSelectionRange(cursorPos, cursorPos);
    input.focus();
    if(typeof renderCandidates === 'function') renderCandidates([]);
    if(typeof fbCandidates !== 'undefined') fbCandidates.innerHTML = '';
    return;
  }

  const token = getLastRomanToken(input.value);

  // ✅ 2. Si rien à suggérer, on nettoie
  if(token.length === 0){
    if(typeof renderCandidates === 'function') renderCandidates([]);
    if(typeof fbCandidates !== 'undefined') fbCandidates.innerHTML = '';
    return;
  }

  // ✅ 3. Sinon, on fait la recherche normale
  const cands = await CandidateProvider.suggestRomanToKhmer(token);
  if(typeof renderCandidates === 'function') renderCandidates(cands);
});

// branchements
(function initCandidateBar(){
  // Attach to all relevant inputs, regardless of social demo
  const inputs = [
    document.getElementById('fbInput'),
    document.getElementById('latinInput'),
    ...Array.from(document.querySelectorAll('textarea, input[type="text"]'))
  ].filter(Boolean);
  inputs.forEach(input => {
    input.addEventListener('input', onInputChanged);
    input.addEventListener('keyup', (e)=>{
      // Enter valide le candidat sélectionné
      if(e.key === 'Enter' && candidateState.items.length){
        commitCandidateToInput(candidateState.items[candidateState.selectedIndex]);
        e.preventDefault();
      }
    });
  });

  // 2) navigation via ←/→ quand la barre a le focus (ou global si tu préfères)
  window.addEventListener('keydown', (e)=>{
    // Only navigate candidates if there are visible items, otherwise allow normal cursor movement
    if(!candidateState.items.length || !candidateState.items.some(Boolean)) return;
    if(e.key === 'ArrowRight'){
      candidateState.selectedIndex = Math.min(candidateState.selectedIndex+1, candidateState.items.length-1);
      updateSelectedCandidate(); e.preventDefault();
    }else if(e.key === 'ArrowLeft'){
      candidateState.selectedIndex = Math.max(candidateState.selectedIndex-1, 0);
      updateSelectedCandidate(); e.preventDefault();
    }
  });

  // 3) expose une fonction pour le clavier virtuel : quand l’utilisateur tape une lettre
  window.fbKeyboardOnCharInserted = ()=> onInputChanged();

  // 4) initial
  if(typeof renderCandidates === 'function') renderCandidates([]);
})();

