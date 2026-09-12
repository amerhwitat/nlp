import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ResultsTable from './ResultsTable'


describe('ResultsTable', () => {
  it('renders Old North Arabian text and transliteration', () => {
    render(<ResultsTable results={[{ id: '1', text: '𐪀𐪁', transliteration: 'hl', confidence: 0.91, source: 'sample.txt', language: 'Old North Arabian', script_variant: 'Dadanitic', codepoints: [0x10A80, 0x10A81] }]} />)
    expect(screen.getByText('𐪀𐪁')).toBeInTheDocument()
    expect(screen.getByText('hl')).toBeInTheDocument()
  })
})
