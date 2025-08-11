/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unused-vars */

import React, { useState, useEffect } from 'react'
import axios from 'axios'

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
} from '@mui/material'
import { ThemeProvider, createTheme, responsiveFontSizes } from '@mui/material/styles'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import ScienceIcon from '@mui/icons-material/Science'

interface AnalysisResult {
  prediction: 'Malignant' | 'Benign'
  probabilities: { Benign: number; Malignant: number }
  llm_report: string
  images: { original: string; grad_cam: string; shap: string }
}

let theme = createTheme({
  palette: {
    primary: {
      main: '#0d47a1',
    },
    secondary: {
      main: '#c62828',
    },
    background: {
      default: '#f4f6f8',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '2.8rem',
      letterSpacing: '-0.5px',
    },
    h5: {
      fontWeight: 600,
    },
    body1: {
      lineHeight: 1.7,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
        },
      },
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
  const [jobId, setJobId] = useState<string | null>(null)

  useEffect(() => {
    if (!jobId || !loading) return
    const intervalId = setInterval(async () => {
      try {
        const { data } = await axios.get(`http://localhost:8000/status/${jobId}`)
        if (data.status === 'complete') {
          setResult(data.result)
          setLoading(false)
          setJobId(null)
          clearInterval(intervalId)
        } else if (data.status === 'failed') {
          setError(`Analysis failed on the server: ${data.error}`)
          setLoading(false)
          setJobId(null)
          clearInterval(intervalId)
        }
      } catch (err) {
        setError('Failed to get analysis status from the server.')
        setLoading(false)
        setJobId(null)
        clearInterval(intervalId)
      }
    }, 5000)
    return () => clearInterval(intervalId)
  }, [jobId, loading])

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setPreview(URL.createObjectURL(file))
      setResult(null)
      setError('')
      setJobId(null)
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image file first.')
      return
    }
    setLoading(true)
    setError('')
    setResult(null)
    const formData = new FormData()
    formData.append('file', selectedFile)
    try {
      const response = await axios.post('http://localhost:8000/analyze', formData)
      setJobId(response.data.job_id)
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to start the analysis job.'
      setError(errorMessage)
      setLoading(false)
    }
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ width: '100vw', minHeight: '100vh', margin: 0, padding: 0 }}>
        <AppBar
          position='static'
          color='default'
          elevation={1}
          sx={{ backgroundColor: 'white', width: '100%' }}
        >
          <Toolbar sx={{ width: '100%', px: 3 }}>
            <ScienceIcon color='primary' sx={{ mr: 2, fontSize: '2rem' }} />
            <Typography variant='h6' component='div' sx={{ flexGrow: 1, fontWeight: 600 }}>
              LLM-Enhanced XAI for Dermatological Classification
            </Typography>
          </Toolbar>
        </AppBar>

        <Box sx={{ width: '100%', px: 3, py: 4 }}>
          <Box sx={{ textAlign: 'center', mb: 5 }}>
            <Typography variant='h4' component='h1' gutterBottom fontWeight={700}>
              Analysis Workbench
            </Typography>
            <Typography variant='subtitle1' color='text.secondary'>
              Upload a dermatological image to generate a prediction and an LLM-powered explanatory
              report.
            </Typography>
          </Box>

          <Paper
            variant='outlined'
            sx={{ p: 3, mb: 5, borderRadius: 4, borderColor: 'divider', width: '100%' }}
          >
            <Grid container spacing={2} alignItems='center'>
              <Grid item xs={12} sm={7}>
                <Button
                  component='label'
                  variant='outlined'
                  fullWidth
                  startIcon={<CloudUploadIcon />}
                  sx={{ py: 1.5 }}
                >
                  {selectedFile ? selectedFile.name : 'Choose an Image File...'}
                  <input
                    type='file'
                    hidden
                    onChange={handleFileChange}
                    accept='image/jpeg, image/png'
                  />
                </Button>
              </Grid>
              <Grid item xs={12} sm={5}>
                <Button
                  variant='contained'
                  size='large'
                  fullWidth
                  disabled={!selectedFile || loading}
                  onClick={handleAnalyze}
                  sx={{ py: 1.5 }}
                >
                  {loading ? <CircularProgress size={24} color='inherit' /> : 'Run Analysis'}
                </Button>
              </Grid>
            </Grid>
            {error && (
              <Alert severity='error' sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
          </Paper>

          {loading && (
            <Card sx={{ width: '100%' }}>
              <CardContent>
                <Typography variant='h5' align='center' gutterBottom>
                  Analysis in Progress...
                </Typography>
                <Typography align='center' color='text.secondary' sx={{ mb: 2 }}>
                  This may take several minutes. Please keep this window open.
                </Typography>
                <LinearProgress variant='indeterminate' />
              </CardContent>
            </Card>
          )}

          {!loading && preview && !result && (
            <Card sx={{ width: '100%' }}>
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
                    style={{ maxHeight: '400px', maxWidth: '100%', borderRadius: '8px' }}
                  />
                </Box>
              </CardContent>
            </Card>
          )}

          {!loading && result && (
            <Grid container spacing={4} sx={{ width: '100%' }}>
              <Grid item xs={12} md={4}>
                <Card sx={{ height: '100%' }}>
                  <CardContent sx={{ textAlign: 'center', p: 3 }}>
                    <Typography color='text.secondary' gutterBottom>
                      Model Prediction
                    </Typography>
                    <Typography
                      variant='h3'
                      component='div'
                      color={result.prediction === 'Malignant' ? 'secondary' : '#2e7d32'}
                      sx={{ fontWeight: 'bold', my: 1 }}
                    >
                      {result.prediction}
                    </Typography>
                    <Typography variant='h6'>
                      {(result.probabilities[result.prediction] * 100).toFixed(2)}%
                      <Typography component='span' color='text.secondary'>
                        {' '}
                        Confidence
                      </Typography>
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={8}>
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    <Typography variant='h5' component='div' gutterBottom>
                      LLM Generated Report
                    </Typography>
                    <Typography variant='body1' component='div' sx={{ whiteSpace: 'pre-wrap' }}>
                      {result.llm_report}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant='h5' component='div' gutterBottom>
                      Visual Explanations (XAI)
                    </Typography>
                    <Grid container spacing={2} justifyContent='center' alignItems='stretch'>
                      {(Object.keys(result.images) as Array<keyof typeof result.images>).map(
                        (key) => (
                          <Grid item xs={12} sm={4} key={key}>
                            <Paper variant='outlined' sx={{ p: 1, height: '100%' }}>
                              <Typography
                                variant='subtitle1'
                                align='center'
                                gutterBottom
                                fontWeight={500}
                              >
                                {key.replace('_', ' ').toUpperCase()}
                              </Typography>
                              <CardMedia
                                component='img'
                                image={result.images[key]}
                                alt={key}
                                sx={{ borderRadius: 1 }}
                              />
                            </Paper>
                          </Grid>
                        )
                      )}
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}

          <Box component='footer' sx={{ mt: 8, py: 3, textAlign: 'center', width: '100%' }}>
            <Typography variant='body2' color='text.secondary'>
              MSc AI Thesis Project | Dangol, Iskierka, and Yildirim | UWE Bristol
            </Typography>
            <Typography variant='caption' color='text.secondary'>
              This tool is for research and educational purposes only and is not intended for
              clinical diagnosis.
            </Typography>
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  )
}

export default App
