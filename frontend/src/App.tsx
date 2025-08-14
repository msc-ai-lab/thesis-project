/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useState, useEffect, useRef } from 'react'
import { fetchEventSource } from '@microsoft/fetch-event-source'
import ReactMarkdown from 'react-markdown'

import {
  CssBaseline,
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  CardMedia,
  CircularProgress,
  Alert,
  Paper,
  Grid,
  LinearProgress,
  AppBar,
  Toolbar,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Snackbar,
} from '@mui/material'
import { ThemeProvider, createTheme, responsiveFontSizes } from '@mui/material/styles'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import ScienceIcon from '@mui/icons-material/Science'
import RefreshIcon from '@mui/icons-material/Refresh'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import ErrorIcon from '@mui/icons-material/Error'
import PendingIcon from '@mui/icons-material/Pending'

const MOCK_API = false
const API_URL = 'http://localhost:8000'
const MAX_FILE_SIZE = 10 * 1024 * 1024
const MAX_RETRIES = 3
const RETRY_DELAY = 2000

interface Influencer {
  case_id: string
  influence_score: number
  ground_truth: 'Benign' | 'Malignant'
  prediction: 'Benign' | 'Malignant'
}

interface AnalysisResult {
  prediction: 'Malignant' | 'Benign'
  probabilities: { Benign: number; Malignant: number }
  llm_report: string
  images: { original: string; grad_cam: string; shap: string }
  influencers_top5: Influencer[]
}

interface ProgressUpdate {
  status?: string
  progress?: number
  step?: string
  detail?: string
  heartbeat?: boolean
  heartbeat_count?: number
  error?: string
  data?: AnalysisResult
  complete?: boolean
}

type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

let theme = createTheme({
  palette: {
    primary: { main: '#0d47a1' },
    secondary: { main: '#c62828' },
    background: { default: '#f4f6f8', paper: '#ffffff' },
    success: { main: '#2e7d32' },
    warning: { main: '#ed6c02' },
    error: { main: '#d32f2f' },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: { fontWeight: 700, fontSize: '2.8rem', letterSpacing: '-0.5px' },
    h5: { fontWeight: 600 },
    body1: { lineHeight: 1.7 },
  },
  components: {
    MuiCard: {
      styleOverrides: { root: { borderRadius: 12, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' } },
    },
    MuiButton: {
      styleOverrides: { root: { borderRadius: 8, textTransform: 'none', fontWeight: 600 } },
    },
    MuiChip: {
      styleOverrides: { root: { fontWeight: 500 } },
    },
  },
})
theme = responsiveFontSizes(theme)

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const [progress, setProgress] = useState<string>('')
  const [progressPercent, setProgressPercent] = useState<number>(0)
  const [progressDetail, setProgressDetail] = useState<string>('')
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected')
  const [retryCount, setRetryCount] = useState<number>(0)
  const [showSnackbar, setShowSnackbar] = useState<boolean>(false)
  const [snackbarMessage, setSnackbarMessage] = useState<string>('')

  const abortControllerRef = useRef<AbortController | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort()
      if (preview) {
        URL.revokeObjectURL(preview)
      }
    }
  }, [preview])

  const validateFile = (file: File): string | null => {
    if (!file.type.startsWith('image/')) {
      return 'Please select an image file (JPEG or PNG)'
    }
    if (file.size > MAX_FILE_SIZE) {
      return `File size must be less than ${MAX_FILE_SIZE / (1024 * 1024)}MB`
    }
    return null
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const validationError = validateFile(file)
      if (validationError) {
        setError(validationError)
        setShowSnackbar(true)
        setSnackbarMessage(validationError)
        return
      }

      if (preview) {
        URL.revokeObjectURL(preview)
      }

      setSelectedFile(file)
      setPreview(URL.createObjectURL(file))
      setResult(null)
      setError('')
      setProgress('')
      setProgressPercent(0)
      setProgressDetail('')
      setConnectionStatus('disconnected')
    }
  }

  const resetAnalysis = () => {
    abortControllerRef.current?.abort()
    setLoading(false)
    setProgress('')
    setProgressPercent(0)
    setProgressDetail('')
    setConnectionStatus('disconnected')
    setRetryCount(0)
    setError('')
  }

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image file first.')
      setShowSnackbar(true)
      setSnackbarMessage('No file selected')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)
    setProgress('Initializing analysis...')
    setProgressPercent(0)
    setProgressDetail('')
    setConnectionStatus('connecting')
    setRetryCount(0)

    if (MOCK_API) {
      console.warn('🔧 Using Mock API for Frontend Development')
      await runMockAnalysis()
      return
    }

    await runRealAnalysis()
  }

  const runMockAnalysis = async () => {
    const mockProgressUpdates = [
      { status: 'Preprocessing input...', progress: 14, detail: 'Resizing and normalizing image' },
      { status: 'Making prediction...', progress: 28, detail: 'Running CNN inference' },
      {
        status: 'Generating Grad-CAM explanation...',
        progress: 42,
        detail: 'Computing gradient-based heatmap',
      },
      {
        status: 'Generating SHAP explanation...',
        progress: 56,
        detail: 'This may take a few minutes',
      },
      {
        status: 'Loading dataset for influence functions...',
        progress: 70,
        detail: 'Preparing training data',
      },
      {
        status: 'Calculating influence functions...',
        progress: 84,
        detail: 'This is the longest step',
      },
      {
        status: 'Generating LLM report...',
        progress: 98,
        detail: 'Creating comprehensive analysis',
      },
    ]

    const mockResult: AnalysisResult = {
      prediction: 'Malignant',
      probabilities: { Benign: 0.08, Malignant: 0.92 },
      llm_report: `### Analysis Report

**Prediction**: Malignant (92% confidence)

#### Visual Analysis
The model has identified several concerning features in the uploaded image:
- Asymmetrical border patterns detected in the lesion
- Irregular coloration with multiple shade variations
- Diameter exceeding typical benign threshold

#### Influential Training Examples
The prediction was most influenced by similar malignant cases in the training set, particularly those exhibiting comparable border irregularities and color variegation.

#### Recommendations
Given the high confidence in malignancy (92%), immediate dermatological consultation is strongly recommended for professional evaluation and potential biopsy.

*Note: This is a mock analysis for UI development. In production, this would contain actual model analysis.*`,
      images: {
        original: preview!,
        grad_cam: 'https://via.placeholder.com/384/FF5722/FFFFFF?text=Grad-CAM+Heatmap',
        shap: 'https://via.placeholder.com/384/3F51B5/FFFFFF?text=SHAP+Values',
      },
      influencers_top5: [
        { case_id: 'ISIC_0024306.jpg', influence_score: 0.1532, ground_truth: 'Malignant', prediction: 'Malignant' },
        { case_id: 'ISIC_0024311.jpg', influence_score: 0.1245, ground_truth: 'Malignant', prediction: 'Malignant' },
        { case_id: 'ISIC_0024298.jpg', influence_score: 0.0987, ground_truth: 'Benign', prediction: 'Benign' },
        { case_id: 'ISIC_0024315.jpg', influence_score: 0.0881, ground_truth: 'Malignant', prediction: 'Malignant' },
        { case_id: 'ISIC_0024289.jpg', influence_score: -0.0753, ground_truth: 'Benign', prediction: 'Benign' },
      ],
    }

    setConnectionStatus('connected')

    for (let i = 0; i < mockProgressUpdates.length; i++) {
      await new Promise((resolve) => setTimeout(resolve, 800))
      const update = mockProgressUpdates[i]
      setProgress(update.status)
      setProgressPercent(update.progress)
      setProgressDetail(update.detail)
    }

    await new Promise((resolve) => setTimeout(resolve, 500))
    setResult(mockResult)
    setLoading(false)
    setConnectionStatus('disconnected')
    setProgress('Analysis complete!')
    setProgressPercent(100)
  }

  const runRealAnalysis = async () => {
    const formData = new FormData()
    formData.append('file', selectedFile!)

    const attemptAnalysis = async (attemptNumber: number = 1) => {
      abortControllerRef.current = new AbortController()

      try {
        await fetchEventSource(`${API_URL}/analyze`, {
          method: 'POST',
          body: formData,
          signal: abortControllerRef.current.signal,
          openWhenHidden: true,

          onopen: async (response) => {
            if (response.ok) {
              setConnectionStatus('connected')
              setRetryCount(0)
              console.log('✅ SSE connection established')
              return
            }

            if (response.status >= 400 && response.status < 500) {
              const errorText = await response.text()
              throw new Error(`Client error: ${errorText}`)
            } else if (response.status >= 500) {
              throw new Error(`Server error: ${response.statusText}`)
            }
          },

          onmessage: (event) => {
            try {
              const data: ProgressUpdate = JSON.parse(event.data)

              if (data.heartbeat) {
                console.log(`Heartbeat #${data.heartbeat_count}`)
                return
              }

              if (data.error) {
                setError(data.error)
                setConnectionStatus('error')
                setLoading(false)
                setShowSnackbar(true)
                setSnackbarMessage(`Analysis failed: ${data.error}`)
                abortControllerRef.current?.abort()
                return
              }

              if (data.status) {
                setProgress(data.status)
                if (data.progress !== undefined) {
                  setProgressPercent(Math.round(data.progress))
                }
                if (data.detail) {
                  setProgressDetail(data.detail)
                }
                if (data.step) {
                  console.log(`Step ${data.step}: ${data.status}`)
                }
              }

              if (data.data) {
                const resultData = data.data as AnalysisResult

                if (resultData.prediction && resultData.probabilities && resultData.images) {
                  setResult(resultData)
                  setLoading(false)
                  setConnectionStatus('disconnected')
                  setProgress('Analysis complete!')
                  setProgressPercent(100)
                  setShowSnackbar(true)
                  setSnackbarMessage('Analysis completed successfully!')
                } else {
                  console.error('Invalid result data structure:', resultData)
                  setError('Received invalid data structure from server')
                }
              }
            } catch (e) {
              console.error('Failed to parse SSE message:', e)
            }
          },

          onclose: () => {
            console.log('SSE connection closed')
            setConnectionStatus('disconnected')
            if (loading) {
              setLoading(false)
            }
          },

          onerror: (err) => {
            console.error('SSE error:', err)
            setConnectionStatus('error')

            if (attemptNumber < MAX_RETRIES) {
              const nextAttempt = attemptNumber + 1
              const delay = RETRY_DELAY * Math.pow(2, attemptNumber - 1)

              console.log(`Retrying... Attempt ${nextAttempt}/${MAX_RETRIES} in ${delay}ms`)
              setProgress(`Connection lost. Retrying... (${nextAttempt}/${MAX_RETRIES})`)
              setRetryCount(nextAttempt)

              setTimeout(() => {
                if (loading) {
                  attemptAnalysis(nextAttempt)
                }
              }, delay)
            } else {
              setError(
                'Connection failed after multiple attempts. Please check your connection and try again.'
              )
              setLoading(false)
              setConnectionStatus('error')
              setShowSnackbar(true)
              setSnackbarMessage('Analysis failed: Connection lost')
            }

            throw err
          },
        })
      } catch (error: any) {
        if (error.name !== 'AbortError') {
          console.error('Analysis error:', error)
          setError(error.message || 'An unexpected error occurred')
          setLoading(false)
          setConnectionStatus('error')
        }
      }
    }

    await attemptAnalysis()
  }

  const getConnectionStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected':
        return <CheckCircleIcon fontSize='small' />
      case 'connecting':
        return <PendingIcon fontSize='small' />
      case 'error':
        return <ErrorIcon fontSize='small' />
      default:
        return undefined
    }
  }

  const getConnectionStatusColor = (): 'success' | 'warning' | 'error' | 'default' => {
    switch (connectionStatus) {
      case 'connected':
        return 'success'
      case 'connecting':
        return 'warning'
      case 'error':
        return 'error'
      default:
        return 'default'
    }
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
        <AppBar position='static' color='default' elevation={1} sx={{ backgroundColor: 'white' }}>
          <Toolbar sx={{ px: { xs: 2, md: 4 } }}>
            <ScienceIcon color='primary' sx={{ mr: 2, fontSize: '2rem' }} />
            <Typography variant='h6' component='div' sx={{ flexGrow: 1, fontWeight: 600 }}>
              LLM-Enhanced XAI for Dermatological Classification
            </Typography>
            {MOCK_API && (
              <Chip label='Mock Mode' color='warning' size='small' sx={{ fontWeight: 600 }} />
            )}
          </Toolbar>
        </AppBar>

        <Box sx={{ p: { xs: 2, md: 4 } }}>
          <Box sx={{ textAlign: 'center', mb: 5 }}>
            <Typography variant='h4' component='h1' gutterBottom fontWeight={700}>
              Analysis Workbench
            </Typography>
            <Typography variant='subtitle1' color='text.secondary'>
              Upload a dermatological image to generate a prediction and an LLM-powered explanatory
              report.
            </Typography>
          </Box>

          <Paper variant='outlined' sx={{ p: 3, mb: 5, borderRadius: 4 }}>
            <Grid container spacing={2} alignItems='center'>
              <Grid item xs={12} sm={6}>
                <Button
                  component='label'
                  variant='outlined'
                  fullWidth
                  startIcon={<CloudUploadIcon />}
                  sx={{ py: 1.5 }}
                  disabled={loading}
                >
                  {selectedFile ? selectedFile.name : 'Choose an Image File...'}
                  <input
                    ref={fileInputRef}
                    type='file'
                    hidden
                    onChange={handleFileChange}
                    accept='image/jpeg, image/png'
                  />
                </Button>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Button
                  variant='contained'
                  size='large'
                  fullWidth
                  disabled={!selectedFile || loading}
                  onClick={handleAnalyze}
                  sx={{ py: 1.5 }}
                  startIcon={loading ? <CircularProgress size={20} color='inherit' /> : null}
                >
                  {loading ? 'Analyzing...' : 'Run Analysis'}
                </Button>
              </Grid>
              <Grid item xs={12} sm={2}>
                <Button
                  variant='outlined'
                  size='large'
                  fullWidth
                  disabled={!loading}
                  onClick={resetAnalysis}
                  sx={{ py: 1.5 }}
                  startIcon={<RefreshIcon />}
                >
                  Cancel
                </Button>
              </Grid>
            </Grid>

            {error && (
              <Alert severity='error' sx={{ mt: 2 }} onClose={() => setError('')}>
                {error}
              </Alert>
            )}

            {selectedFile && (
              <Box sx={{ mt: 2 }}>
                <Typography variant='body2' color='text.secondary'>
                  File: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(2)} KB)
                </Typography>
              </Box>
            )}
          </Paper>

          {loading && (
            <Card sx={{ mb: 4 }}>
              <CardContent>
                <Box sx={{ mb: 3 }}>
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      mb: 2,
                    }}
                  >
                    <Typography variant='h5' component='div'>
                      Analysis in Progress...
                    </Typography>
                    <Chip
                      icon={getConnectionStatusIcon()}
                      label={connectionStatus}
                      color={getConnectionStatusColor()}
                      size='small'
                      sx={{ textTransform: 'capitalize' }}
                    />
                  </Box>

                  <Typography color='text.secondary' sx={{ mb: 1, fontStyle: 'italic' }}>
                    {progress || 'Initializing analysis pipeline...'}
                  </Typography>

                  {progressDetail && (
                    <Typography variant='body2' color='text.secondary'>
                      {progressDetail}
                    </Typography>
                  )}

                  {retryCount > 0 && (
                    <Alert severity='warning' sx={{ mt: 2 }}>
                      Connection interrupted. Retry attempt {retryCount} of {MAX_RETRIES}...
                    </Alert>
                  )}
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Box sx={{ width: '100%', mr: 1 }}>
                    <LinearProgress
                      variant={progressPercent > 0 ? 'determinate' : 'indeterminate'}
                      value={progressPercent}
                    />
                  </Box>
                  {progressPercent > 0 && (
                    <Box sx={{ minWidth: 35 }}>
                      <Typography variant='body2' color='text.secondary'>
                        {`${progressPercent}%`}
                      </Typography>
                    </Box>
                  )}
                </Box>

                <Typography
                  variant='caption'
                  color='text.secondary'
                  sx={{ mt: 2, display: 'block' }}
                >
                  This analysis may take several minutes. Please keep this window open.
                </Typography>
              </CardContent>
            </Card>
          )}

          {!loading && preview && !result && (
            <Card>
              <CardContent>
                <Typography variant='h5' component='div' gutterBottom>
                  Image Preview
                </Typography>
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'center',
                    p: 2,
                    bgcolor: 'grey.100',
                    borderRadius: 2,
                  }}
                >
                  <img
                    src={preview}
                    alt='Preview'
                    style={{
                      maxHeight: '400px',
                      maxWidth: '100%',
                      borderRadius: '8px',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          )}

          {!loading && result && result.prediction && (
            <Grid container spacing={4}>
              {/* Prediction Card */}
              <Grid item xs={12} md={4}>
                <Card sx={{ height: '100%', textAlign: 'center' }}>
                  <CardContent sx={{ p: 3 }}>
                    <Typography color='text.secondary' gutterBottom>
                      Model Prediction
                    </Typography>
                    <Typography
                      variant='h3'
                      component='div'
                      color={result.prediction === 'Malignant' ? 'secondary' : 'success.main'}
                      sx={{ fontWeight: 'bold', my: 2 }}
                    >
                      {result.prediction}
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant='h5' component='div'>
                        {result.probabilities && result.probabilities[result.prediction]
                          ? (result.probabilities[result.prediction] * 100).toFixed(1)
                          : 'N/A'}
                        %
                      </Typography>
                      <Typography variant='body2' color='text.secondary'>
                        Confidence Score
                      </Typography>
                    </Box>
                    <Box sx={{ mt: 3, pt: 2, borderTop: 1, borderColor: 'divider' }}>
                      <Typography variant='body2' color='text.secondary' gutterBottom>
                        Probability Distribution
                      </Typography>
                      {result.probabilities && (
                        <Box sx={{ mt: 1 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Typography variant='body2'>Benign:</Typography>
                            <Typography variant='body2' fontWeight={500}>
                              {(result.probabilities.Benign * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant='body2'>Malignant:</Typography>
                            <Typography variant='body2' fontWeight={500}>
                              {(result.probabilities.Malignant * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                        </Box>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              {/* LLM Report Card */}
              <Grid item xs={12} md={8}>
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    <Typography variant='h5' component='div' gutterBottom>
                      LLM Generated Report
                    </Typography>
                    <Paper
                      variant='outlined'
                      sx={{
                        p: 2,
                        bgcolor: 'grey.50',
                        maxHeight: '400px',
                        overflowY: 'auto',
                      }}
                    >
                      <ReactMarkdown
                        components={{
                          h3: ({ node, ...props }) => (
                            <Typography
                              variant='h6'
                              {...props}
                              sx={{ fontWeight: 600, mt: 2, mb: 1 }}
                            />
                          ),
                          p: ({ node, ...props }) => (
                            <Typography variant='body1' paragraph {...props} />
                          ),
                          li: ({ node, ...props }) => (
                            <li>
                              <Typography variant='body1' component='span' {...props} />
                            </li>
                          ),
                          strong: ({ node, ...props }) => (
                            <strong {...props} style={{ fontWeight: 'bold' }} />
                          ),
                        }}
                      >
                        {result.llm_report}
                      </ReactMarkdown>
                    </Paper>
                  </CardContent>
                </Card>
              </Grid>

              {/* Visual Explanations */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant='h5' component='div' gutterBottom>
                      Visual Explanations (XAI)
                    </Typography>
                    <Typography variant='body2' color='text.secondary' sx={{ mb: 2 }}>
                      These visualizations show which parts of the image contributed most to the
                      model's decision.
                    </Typography>
                    <Grid container spacing={2} justifyContent='center' alignItems='stretch'>
                      {result.images &&
                        [
                          { key: 'original', label: 'Original Image' },
                          { key: 'grad_cam', label: 'Grad-CAM Heatmap' },
                          { key: 'shap', label: 'SHAP Values' },
                        ].map(
                          ({ key, label }) =>
                            result.images[key as keyof typeof result.images] && (
                              <Grid item xs={12} sm={6} md={4} key={key}>
                                <Paper
                                  variant='outlined'
                                  sx={{
                                    p: 2,
                                    height: '100%',
                                    transition: 'transform 0.2s',
                                    '&:hover': { transform: 'scale(1.02)' },
                                  }}
                                >
                                  <Typography
                                    variant='subtitle1'
                                    align='center'
                                    gutterBottom
                                    fontWeight={500}
                                  >
                                    {label}
                                  </Typography>
                                  <CardMedia
                                    component='img'
                                    image={result.images[key as keyof typeof result.images]}
                                    alt={label}
                                    sx={{
                                      borderRadius: 1,
                                      boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                                    }}
                                  />
                                </Paper>
                              </Grid>
                            )
                        )}
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>

              {/* Influential Training Images */}
              {result.influencers_top5 && result.influencers_top5.length > 0 && (
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant='h5' component='div' gutterBottom>
                        Top 5 Influential Training Images
                      </Typography>
                      <Typography variant='body2' color='text.secondary' sx={{ mb: 2 }}>
                        These training images had the most influence on the model's prediction for
                        your uploaded image.
                      </Typography>
                      <TableContainer component={Paper} variant='outlined'>
                        <Table sx={{ minWidth: 650 }} aria-label='influential images table'>
                          <TableHead>
                            <TableRow sx={{ bgcolor: 'grey.50' }}>
                              <TableCell sx={{ fontWeight: 600 }}>Training Case ID</TableCell>
                              <TableCell align='right' sx={{ fontWeight: 600 }}>
                                Influence Score
                              </TableCell>
                              <TableCell sx={{ fontWeight: 600 }}>Ground Truth</TableCell>
                              <TableCell sx={{ fontWeight: 600 }}>Impact</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {result.influencers_top5.map((row, index) => (
                              <TableRow
                                key={row.case_id || index}
                                sx={{ '&:nth-of-type(odd)': { bgcolor: 'action.hover' } }}
                              >
                                <TableCell component='th' scope='row'>
                                  <Typography variant='body2' fontWeight={500}>
                                    {row.case_id} {/* Use case_id */}
                                  </Typography>
                                </TableCell>
                                <TableCell align='right'>
                                  <Typography
                                    variant='body2'
                                    color={row.influence_score > 0 ? 'success.main' : 'error.main'} // Use influence_score
                                    fontWeight={500}
                                  >
                                    {row.influence_score > 0 ? '+' : ''}
                                    {row.influence_score.toFixed(4)} {/* Use influence_score */}
                                  </Typography>
                                </TableCell>
                                <TableCell>
                                  <Chip
                                    label={row.ground_truth}
                                    size='small'
                                    color={
                                      row.ground_truth === 'Malignant' ? 'secondary' : 'success'
                                    }
                                    sx={{ fontWeight: 500 }}
                                  />
                                </TableCell>
                                <TableCell>
                                  <Typography variant='body2' color='text.secondary'>
                                    {row.influence_score > 0 ? 'Supports' : 'Contradicts'}{' '}
                                    Prediction {/* Use influence_score */}
                                  </Typography>
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {/* Action Buttons */}
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
                  <Button
                    variant='contained'
                    onClick={() => {
                      setSelectedFile(null)
                      setPreview(null)
                      setResult(null)
                      setProgress('')
                      setProgressPercent(0)
                      if (fileInputRef.current) {
                        fileInputRef.current.value = ''
                      }
                    }}
                  >
                    Analyze Another Image
                  </Button>
                  <Button
                    variant='outlined'
                    onClick={() => {
                      const reportData = {
                        prediction: result.prediction,
                        probabilities: result.probabilities,
                        report: result.llm_report,
                        influencers: result.influencers_top5,
                      }
                      const blob = new Blob([JSON.stringify(reportData, null, 2)], {
                        type: 'application/json',
                      })
                      const url = URL.createObjectURL(blob)
                      const a = document.createElement('a')
                      a.href = url
                      a.download = `analysis_report_${new Date().toISOString()}.json`
                      a.click()
                      URL.revokeObjectURL(url)
                    }}
                  >
                    Download Report
                  </Button>
                </Box>
              </Grid>
            </Grid>
          )}

          <Box component='footer' sx={{ mt: 8, py: 3, textAlign: 'center' }}>
            <Typography variant='body2' color='text.secondary'>
              MSc AI Thesis Project | Dangol, Iskierka, and Yildirim | UWE Bristol
            </Typography>
            <Typography variant='caption' color='text.secondary'>
              This tool is for research and educational purposes only and is not intended for
              clinical diagnosis.
            </Typography>
          </Box>
        </Box>

        {/* Snackbar for notifications */}
        <Snackbar
          open={showSnackbar}
          autoHideDuration={6000}
          onClose={() => setShowSnackbar(false)}
          message={snackbarMessage}
        />
      </Box>
    </ThemeProvider>
  )
}

export default App
